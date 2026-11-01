from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

from saife.events import Sink
from saife.guard import Guard
from saife.server.app import create_app
from saife.types import Event, Report, RunUsage, Source


def _report(run_id: str, answer: str) -> Report:
    return Report(
        run_id=run_id,
        status="CAUTION",
        preset="fast",
        original_draft=answer,
        final_answer=answer,
        claims=[],
        passes=[],
        repairs=[],
        alerts=[],
        usage=RunUsage(),
        risk=0.4,
        stop_cause="repair_disabled",
    )


class FakeGuard:
    """Emits a fixed event sequence to the per-run sinks, like the real Guard."""

    def __init__(self, preset: str, *, fail: bool = False) -> None:
        self.preset = preset
        self.fail = fail
        self.calls: list[dict[str, Any]] = []

    async def _emit(self, run_id: str, sinks: Sequence[Sink], types: list[str]) -> None:
        for seq, t in enumerate(types):
            await sinks[0](Event(type=t, run_id=run_id, seq=seq, ts=time.time()))  # type: ignore[arg-type]

    async def check(
        self,
        answer: str,
        sources: list[Source] | None = None,
        question: str | None = None,
        *,
        run_id: str | None = None,
        extra_sinks: Sequence[Sink] = (),
    ) -> Report:
        assert run_id is not None
        self.calls.append({"mode": "check", "answer": answer, "sources": sources})
        if self.fail:
            await self._emit(run_id, extra_sinks, ["run.started"])
            raise RuntimeError("boom")
        report = _report(run_id, answer)
        await self._emit(run_id, extra_sinks, ["run.started", "pass.started", "run.completed"])
        return report

    async def run(
        self,
        question: str,
        sources: list[Source] | None = None,
        *,
        run_id: str | None = None,
        extra_sinks: Sequence[Sink] = (),
    ) -> Report:
        assert run_id is not None
        self.calls.append({"mode": "run", "question": question})
        await self._emit(run_id, extra_sinks, ["run.started", "run.completed"])
        return _report(run_id, "generated")


def _client(tmp_path: Path, **kw: Any) -> tuple[TestClient, list[FakeGuard]]:
    made: list[FakeGuard] = []

    def factory(preset: str) -> Guard:
        g = FakeGuard(preset, **kw)
        made.append(g)
        return cast(Guard, g)

    traps = tmp_path / "traps.jsonl"
    traps.write_text(
        json.dumps({"id": "t1", "category": "citation", "question": "q?", "expected_flag": "x"})
        + "\n",
        encoding="utf-8",
    )
    return TestClient(create_app(factory, traps_path=traps)), made


def _stream(client: TestClient, payload: dict[str, Any]) -> list[dict[str, Any]]:
    """POST a run and read its SSE stream; returns the events in order."""
    events = []
    with client.stream("POST", "/api/runs/stream", json=payload) as r:
        assert r.status_code == 200
        for line in r.iter_lines():
            if line.startswith("data:"):
                events.append(json.loads(line.split(":", 1)[1]))
    return events


def test_check_run_streams_events_in_one_request(tmp_path: Path) -> None:
    client, made = _client(tmp_path)
    with client:
        events = _stream(
            client,
            {
                "answer": "The sky is green.",
                "preset": "fast",
                "sources": [{"id": "s", "text": "t"}],
            },
        )
    assert [e["type"] for e in events] == ["run.started", "pass.started", "run.completed"]
    assert len({e["run_id"] for e in events}) == 1
    assert made[0].preset == "fast" and made[0].calls[0]["mode"] == "check"
    assert made[0].calls[0]["sources"][0].id == "s"


def test_question_only_uses_generate_mode(tmp_path: Path) -> None:
    client, made = _client(tmp_path)
    with client:
        assert _stream(client, {"question": "Who?"})[-1]["type"] == "run.completed"
    assert made[0].calls[0] == {"mode": "run", "question": "Who?"}
    assert made[0].preset == "balanced"


def test_guard_crash_still_terminates_stream(tmp_path: Path) -> None:
    client, _ = _client(tmp_path, fail=True)
    with client:
        events = _stream(client, {"answer": "x"})
    assert [e["type"] for e in events] == ["run.started", "run.failed"]
    assert events[-1]["data"]["error"].startswith("RuntimeError")


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"question": "  "},
        {"answer": "x", "preset": "yolo"},
        {"answer": "x", "preset": "strict"},  # needs web search: not offered in the showcase
        {"answer": "x" * 40_001},
    ],
)
def test_invalid_requests_rejected(tmp_path: Path, payload: dict[str, Any]) -> None:
    client, made = _client(tmp_path)
    with client:
        assert client.post("/api/runs/stream", json=payload).status_code == 422
    assert not made


def test_oversized_body_rejected(tmp_path: Path) -> None:
    client, _ = _client(tmp_path)
    with client:
        r = client.post(
            "/api/runs/stream",
            content=b"{" + b" " * 1_000_001 + b"}",
            headers={"content-type": "application/json"},
        )
        assert r.status_code == 413


def test_health(tmp_path: Path) -> None:
    client, _ = _client(tmp_path)
    with client:
        assert client.get("/api/health").json() == {"status": "ok"}
        assert client.get("/").status_code == 404  # API only: the UI is on Vercel


def test_not_configured_is_501_without_spending_quota(tmp_path: Path) -> None:
    from saife.errors import ConfigError

    def factory(preset: str) -> Guard:
        raise ConfigError("no key")

    with TestClient(create_app(factory, traps_path=tmp_path / "t.jsonl")) as client:
        assert client.post("/api/runs/stream", json={"answer": "x"}).status_code == 501


def test_rate_limited_run_is_429_before_any_model_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi import HTTPException

    import saife.server.app as server

    seen: list[str] = []

    def limited(request: Any, bucket: str) -> None:
        seen.append(bucket)
        raise HTTPException(status_code=429, detail={"queries_used": 3})

    monkeypatch.setattr(server, "check_rate_limit", limited)
    client, made = _client(tmp_path)
    with client:
        r = client.post("/api/runs/stream", json={"question": "Who?"})
        assert r.status_code == 429 and r.json()["detail"]["queries_used"] == 3
        assert client.post("/api/runs/stream", json={"answer": "x"}).status_code == 429
    assert seen == ["ask", "check"]
    assert all(not g.calls for g in made)


def test_presets_and_traps(tmp_path: Path) -> None:
    client, _ = _client(tmp_path)
    with client:
        presets = client.get("/api/presets").json()
        assert set(presets) == {"fast", "balanced"}
        traps = client.get("/api/traps").json()
        assert traps == [{"id": "t1", "category": "citation", "question": "q?"}]


def test_cors_only_dev_origin(tmp_path: Path) -> None:
    client, _ = _client(tmp_path)
    with client:
        ok = client.get("/api/presets", headers={"Origin": "http://localhost:5173"})
        bad = client.get("/api/presets", headers={"Origin": "https://evil.example"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert "access-control-allow-origin" not in bad.headers


def test_cors_origins_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORS_ORIGINS", "https://saife-demo.example, https://other.example/")
    client, _ = _client(tmp_path)
    with client:
        ok = client.get("/api/presets", headers={"Origin": "https://other.example"})
        dev = client.get("/api/presets", headers={"Origin": "http://localhost:5173"})
    assert ok.headers.get("access-control-allow-origin") == "https://other.example"
    assert "access-control-allow-origin" not in dev.headers
