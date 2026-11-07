"""FastAPI server: run a check and stream its events over SSE in the same request.

``uv run uvicorn saife.server.app:app`` serves the API on :8000. The demo UI is a separate
Vite app (``web/``); this server answers ``/api/*`` only.

Each run is self-contained: ``POST /api/runs/stream`` runs the guard inside the request and
streams every event, ending with ``run.completed`` (which carries the report) or
``run.failed``. Nothing is kept on the server between requests, so any instance can answer
and scale-to-zero loses nothing.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator
from sse_starlette import EventSourceResponse, ServerSentEvent

from saife.config import PRESETS
from saife.errors import ConfigError
from saife.guard import Guard
from saife.server.rate_limit import check_rate_limit
from saife.types import Event, Source

log = logging.getLogger(__name__)

TERMINAL = frozenset({"run.completed", "run.failed"})
MAX_BODY_BYTES = 1_000_000
MAX_ACTIVE_RUNS = 8  # per instance; more → 423 (busy)
DEV_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
# The strict preset needs Anthropic web search, which the showcase does not have.
PUBLIC_PRESETS = ("fast", "balanced")
REPO_ROOT = Path(__file__).resolve().parents[3]
TRAPS_PATH = REPO_ROOT / "bench" / "datasets" / "traps" / "traps.jsonl"
RESULTS_DIR = REPO_ROOT / "bench" / "results"
MAX_RESULT_BYTES = 5_000_000

GuardFactory = Callable[[str], Guard]


class SourceIn(BaseModel):
    id: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=100_000)
    title: str | None = Field(default=None, max_length=500)
    url: str | None = Field(default=None, max_length=2000)


class RunRequest(BaseModel):
    question: str | None = Field(default=None, max_length=4000)
    answer: str | None = Field(default=None, max_length=40_000)
    sources: list[SourceIn] = Field(default_factory=list, max_length=20)
    preset: Literal["fast", "balanced"] = "balanced"

    @model_validator(mode="after")
    def _needs_input(self) -> RunRequest:
        if not (self.question or "").strip() and not (self.answer or "").strip():
            raise ValueError("provide a question, an answer, or both")
        return self

    @property
    def is_check(self) -> bool:
        return bool((self.answer or "").strip())


def _default_factory(preset: str) -> Guard:
    return Guard(preset)


def _cors_origins() -> list[str]:
    raw = os.getenv("CORS_ORIGINS", "")
    origins = [o.strip().rstrip("/") for o in raw.split(",") if o.strip()]
    return origins or DEV_ORIGINS


def _load_traps(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    keep = ("id", "category", "question", "sources", "notes")
    traps = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            traps.append({k: item[k] for k in keep if k in item})
    return traps


def _load_results(results_dir: Path) -> list[dict[str, Any]]:
    """Benchmark summaries (``meta`` + ``metrics``), newest first. The listing takes no
    user input, and only regular files directly inside ``results_dir`` are read."""
    if not results_dir.is_dir():
        return []
    root = results_dir.resolve()
    found: list[tuple[str, float, dict[str, Any]]] = []
    for path in root.glob("*.json"):
        try:
            if path.is_symlink() or not path.is_file() or path.resolve().parent != root:
                continue
            stat = path.stat()
            if stat.st_size > MAX_RESULT_BYTES:
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        meta, metrics = data.get("meta"), data.get("metrics")
        if not isinstance(meta, dict) or not isinstance(metrics, dict):
            continue
        summary = {"name": path.stem, "meta": meta, "metrics": metrics}
        found.append((str(meta.get("date", "")), stat.st_mtime, summary))
    found.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return [summary for _, _, summary in found]


def create_app(
    guard_factory: GuardFactory = _default_factory,
    *,
    traps_path: Path = TRAPS_PATH,
    results_dir: Path = RESULTS_DIR,
) -> FastAPI:
    app = FastAPI(title="sAIfe", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    active = {"runs": 0}

    @app.middleware("http")
    async def limit_body(request: Request, call_next: Callable[[Request], Any]) -> Response:
        length = request.headers.get("content-length")
        if length is not None and (not length.isdigit() or int(length) > MAX_BODY_BYTES):
            return JSONResponse({"detail": "request body too large"}, status_code=413)
        response: Response = await call_next(request)
        return response

    async def execute(
        guard: Guard, req: RunRequest, run_id: str, out: asyncio.Queue[Event]
    ) -> None:
        sources = [Source(**s.model_dump()) for s in req.sources] or None

        async def sink(event: Event) -> None:
            await out.put(event)

        try:
            if req.is_check:
                assert req.answer is not None
                await guard.check(
                    req.answer,
                    sources=sources,
                    question=req.question,
                    run_id=run_id,
                    extra_sinks=[sink],
                )
            else:
                assert req.question is not None
                await guard.run(req.question, sources=sources, run_id=run_id, extra_sinks=[sink])
        except Exception as exc:  # the stream must still terminate (fail closed)
            log.exception("run %s failed", run_id)
            await out.put(
                Event(
                    type="run.failed",
                    run_id=run_id,
                    seq=10**9,  # after anything already sent; the client only needs order
                    ts=time.time(),
                    data={"error": f"{type(exc).__name__}: {exc}"},
                )
            )

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/runs/stream")
    async def run_stream(req: RunRequest, request: Request) -> EventSourceResponse:
        if active["runs"] >= MAX_ACTIVE_RUNS:
            raise HTTPException(status_code=423, detail="the demo is busy; try again shortly")
        try:
            guard = guard_factory(req.preset)
        except ConfigError as exc:
            raise HTTPException(status_code=501, detail=f"AI is not configured: {exc}") from exc
        # One hit per run, right before the model calls start (after validation and config).
        check_rate_limit(request, "check" if req.is_check else "ask")
        run_id = os.urandom(6).hex()
        queue: asyncio.Queue[Event] = asyncio.Queue()
        active["runs"] += 1

        async def gen() -> AsyncIterator[ServerSentEvent]:
            task = asyncio.create_task(execute(guard, req, run_id, queue))
            try:
                while True:
                    event = await queue.get()
                    yield ServerSentEvent(
                        data=event.model_dump_json(), event=event.type, id=str(event.seq)
                    )
                    if event.type in TERMINAL:
                        return
            finally:
                task.cancel()  # client went away: stop spending model calls
                active["runs"] -= 1

        return EventSourceResponse(gen(), ping=15)

    @app.get("/api/presets")
    async def presets() -> dict[str, Any]:
        return {
            name: PRESETS[name].model_dump(mode="json", exclude={"roles"})
            for name in PUBLIC_PRESETS
        }

    @app.get("/api/traps")
    async def traps() -> list[dict[str, Any]]:
        return _load_traps(traps_path)

    @app.get("/api/results")
    def results() -> list[dict[str, Any]]:
        return _load_results(results_dir)

    return app


app = create_app()
