"""Benchmark runner, CLI ``bench`` command and ``GET /api/results``, all offline (FakeLLM)."""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

import httpx2
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from bench.run import BenchOptions, CostLimitExceeded, TrapMetrics, build_config, run_bench
from bench.schema import TRAPS_PATH
from saife import cli
from saife.config import ANTHROPIC, VERIFIER, VERIFIER_ROLES, GuardConfig, default_roles
from saife.guard import Guard
from saife.llm.base import Usage
from saife.llm.fake import FakeLLM, FakeRequest, FakeResponse
from saife.llm.router import Router
from saife.server.app import create_app
from saife.types import Report

USAGE = Usage(input_tokens=1000, output_tokens=200)
SENTENCE = re.compile(r"(?<=[.!?])\s+")
SOURCE_BLOCK = re.compile(r'<source id="([^"]+)">\n(.*?)\n</source>', re.S)


def _user(req: FakeRequest) -> str:
    return html.unescape(str(req.messages[-1]["content"]))


def handler(req: FakeRequest) -> FakeResponse:
    """One claim per sentence; a claim is SUPPORTED only if it appears word for word in a
    source (so the quote verifies), otherwise NOT_FOUND."""
    content = _user(req)
    if req.schema_name == "DecomposeOut":
        draft = content.split("<draft>", 1)[1].split("</draft>", 1)[0].strip()
        claims = [
            {"text": s, "anchor": s, "kind": "entity", "checkworthy": True}
            for s in SENTENCE.split(draft)
            if s.strip()
        ]
        return FakeResponse(text=json.dumps({"claims": claims}), usage=USAGE)
    if req.schema_name == "GroundingJudgement":
        quote = content.split("<claim>", 1)[1].split("</claim>", 1)[0].strip().rstrip(".")
        sources = SOURCE_BLOCK.findall(content)
        hit = next((sid for sid, text in sources if quote and quote in text), None)
        if hit is not None:
            quotes = [{"source_id": hit, "quote": quote}]
            body: dict[str, Any] = {"verdict": "SUPPORTED", "confidence": 0.9, "quotes": quotes}
        else:
            body = {"verdict": "NOT_FOUND", "confidence": 0.6, "quotes": []}
        return FakeResponse(text=json.dumps(body), usage=USAGE)
    if req.kind == "complete":
        return FakeResponse(text="The kettle was invented in 1999 by Ada Lovelace.", usage=USAGE)
    raise AssertionError(f"unexpected call {req.schema_name}")


class Env:
    def __init__(self, tmp_path: Path) -> None:
        self.tmp = tmp_path
        self.claude = FakeLLM(ANTHROPIC, handler=handler)
        self.verifier = FakeLLM(VERIFIER, handler=handler)

    def router(self, config: GuardConfig) -> Router:
        clients = {ANTHROPIC: self.claude, VERIFIER: self.verifier}
        return Router(clients, default_roles(), policy=config.fallback_policy)

    def extras(self) -> dict[str, Any]:
        async def resolver(host: str) -> list[str]:
            return ["93.184.216.34"]

        transport = httpx2.MockTransport(lambda r: httpx2.Response(200))
        return {"resolver": resolver, "http_client": httpx2.AsyncClient(transport=transport)}

    def factory(self, config: GuardConfig, trace_dir: Path) -> Guard:
        return Guard(
            config, router=self.router(config), trace_dir=self.tmp / "traces", **self.extras()
        )

    def opts(self, **kw: Any) -> BenchOptions:
        base: dict[str, Any] = {
            "set": "seeded",
            "preset": "fast",
            "limit": 3,
            "out": self.tmp / "results",
            "date": "2026-10-10",
        }
        return BenchOptions(**{**base, **kw})

    def schemas(self, llm: FakeLLM) -> list[str | None]:
        return [r.schema_name for r in llm.calls]


@pytest.fixture
def env(tmp_path: Path) -> Env:
    return Env(tmp_path)


async def test_seeded_run_end_to_end(env: Env) -> None:
    logs: list[str] = []
    result = await run_bench(env.opts(), env.factory, log=logs.append)

    name = "2026-10-10-seeded-fast"
    assert result.meta.run_name == name and result.meta.n_items == 3
    assert result.meta.detectors == ["grounding", "deterministic"]
    assert result.meta.models["grounding_judge"].startswith("verifier/")
    assert result.metrics.items == 3 and result.metrics.silent_failures == 0
    assert result.meta.total_cost_usd > 0
    assert [i["id"] for i in result.items] == ["seeded-p01", "seeded-p02", "seeded-p03"]
    keys = {"status", "flagged", "residual_before", "residual_after"}
    assert all(keys <= i.keys() for i in result.items)
    assert any("cost estimate" in line for line in logs)

    out = env.tmp / "results"
    saved = json.loads((out / f"{name}.json").read_text(encoding="utf-8"))
    assert set(saved) == {"meta", "metrics", "items"}
    assert saved["meta"]["set"] == "seeded" and saved["meta"]["verifier"] == "cross"
    md = (out / f"{name}.md").read_text(encoding="utf-8")
    assert "Silent failures: 0" in md and "Residual hallucination rate" in md
    for item in ("seeded-p01", "seeded-p02", "seeded-p03"):
        Report.model_validate_json((out / name / "items" / f"{item}.json").read_text("utf-8"))


async def test_resume_skips_items_already_done(env: Env) -> None:
    await run_bench(env.opts(limit=2), env.factory, log=lambda m: None)
    first = env.schemas(env.claude).count("DecomposeOut")
    logs: list[str] = []
    result = await run_bench(env.opts(limit=3), env.factory, log=logs.append)
    assert env.schemas(env.claude).count("DecomposeOut") - first == 1  # only the new item
    assert any("2 done, 1 to run" in line for line in logs)
    assert result.meta.n_items == 3


async def test_half_written_item_is_rerun(env: Env) -> None:
    items = env.tmp / "results" / "2026-10-10-seeded-fast" / "items"
    items.mkdir(parents=True)
    (items / "seeded-p01.json").write_text("{truncated", encoding="utf-8")
    result = await run_bench(env.opts(limit=1), env.factory, log=lambda m: None)
    assert result.meta.n_items == 1
    Report.model_validate_json((items / "seeded-p01.json").read_text("utf-8"))


async def test_ablation_removes_detector(env: Env) -> None:
    opts = env.opts(ablate=("grounding",), limit=1)
    assert build_config(opts).detectors == ("deterministic",)
    result = await run_bench(opts, env.factory, log=lambda m: None)
    assert result.meta.run_name == "2026-10-10-seeded-fast-ablate-grounding"
    assert result.meta.ablate == ["grounding"] and result.meta.detectors == ["deterministic"]
    assert "GroundingJudgement" not in env.schemas(env.claude) + env.schemas(env.verifier)


def test_ablating_unknown_or_every_detector_is_rejected(env: Env) -> None:
    with pytest.raises(ValueError, match="cannot ablate"):
        build_config(env.opts(ablate=("web_evidence",)))
    with pytest.raises(ValueError, match="every detector"):
        build_config(env.opts(ablate=("grounding", "deterministic")))


async def test_verifier_same_routes_every_judge_to_claude(env: Env) -> None:
    opts = env.opts(verifier="same", limit=1)
    cfg = build_config(opts)
    assert cfg.roles is not None
    assert all(cfg.roles[r].provider == ANTHROPIC for r in VERIFIER_ROLES)
    assert cfg.roles[next(iter(VERIFIER_ROLES))].model == "claude-opus-5"

    result = await run_bench(opts, env.factory, log=lambda m: None)
    assert result.meta.run_name.endswith("-same")
    assert env.verifier.calls == []
    assert "GroundingJudgement" in env.schemas(env.claude)
    for role in ("grounding_judge", "consistency_judge", "web_judge"):
        assert result.meta.models[role] == "anthropic/claude-opus-5"


async def test_cost_guard_aborts_after_estimate(env: Env) -> None:
    with pytest.raises(CostLimitExceeded) as info:
        await run_bench(env.opts(limit=8, max_cost=1e-9), env.factory, log=lambda m: None)
    assert info.value.projected > 1e-9
    items = env.tmp / "results" / "2026-10-10-seeded-fast" / "items"
    assert len(list(items.glob("*.json"))) == 5  # the estimate batch is kept for resume
    assert not (env.tmp / "results" / "2026-10-10-seeded-fast.json").exists()


async def test_traps_report_flag_rate(env: Env) -> None:
    result = await run_bench(env.opts(set="traps", limit=2), env.factory, log=lambda m: None)
    assert isinstance(result.metrics, TrapMetrics)
    assert result.metrics.items == 2 and result.metrics.flag_rate == 1.0  # NOT_FOUND → flagged
    assert "complete" in [r.kind for r in env.claude.calls]  # Guard.run generated the answers
    assert all("category" in i for i in result.items)
    md = (env.tmp / "results" / "2026-10-10-traps-fast.md").read_text(encoding="utf-8")
    assert "Trap flag rate" in md
    assert TRAPS_PATH.exists()


# --- CLI -------------------------------------------------------------------------------

runner = CliRunner()


@pytest.fixture
def cli_env(env: Env, monkeypatch: pytest.MonkeyPatch) -> Env:
    monkeypatch.setattr(cli, "make_router", env.router)
    monkeypatch.setattr(cli, "guard_kwargs", env.extras)
    monkeypatch.chdir(env.tmp)  # traces go under ./runs
    return env


def _bench_args(env: Env, *extra: str) -> list[str]:
    out = str(env.tmp / "results")
    return ["bench", "--set", "seeded", "-p", "fast", "--limit", "2", "--out", out,
            "--date", "2026-10-10", *extra]  # fmt: skip


def test_cli_bench_passes_with_zero_silent_failures(cli_env: Env) -> None:
    result = runner.invoke(cli.app, _bench_args(cli_env))
    assert result.exit_code == 0, result.output
    assert "SILENT FAILURES: 0" in result.output
    assert (cli_env.tmp / "results" / "2026-10-10-seeded-fast.json").exists()


def test_cli_bench_exits_nonzero_on_silent_failure(cli_env: Env) -> None:
    assert runner.invoke(cli.app, _bench_args(cli_env)).exit_code == 0
    path = cli_env.tmp / "results" / "2026-10-10-seeded-fast" / "items" / "seeded-p01.json"
    report = Report.model_validate_json(path.read_text(encoding="utf-8"))
    # A SUPPORTED claim with no verified evidence: the safety failure the bench must catch.
    bad = report.claims[0].model_copy(update={"status": "SUPPORTED", "verdicts": []})
    path.write_text(
        report.model_copy(update={"claims": [bad, *report.claims[1:]]}).model_dump_json(),
        encoding="utf-8",
    )
    result = runner.invoke(cli.app, _bench_args(cli_env))
    assert result.exit_code == 1, result.output
    assert "SILENT FAILURES: 1" in result.output
    assert f"{report.run_id}:{bad.claim.id}" in result.output


def test_cli_bench_cost_abort_exits_2(cli_env: Env) -> None:
    result = runner.invoke(cli.app, [*_bench_args(cli_env), "--max-cost", "0.000000001"])
    assert result.exit_code == 2, result.output
    assert "aborted" in result.output


def test_cli_bench_rejects_bad_options(cli_env: Env) -> None:
    assert runner.invoke(cli.app, [*_bench_args(cli_env), "--verifier", "x"]).exit_code != 0
    result = runner.invoke(cli.app, [*_bench_args(cli_env), "--ablate", "web_evidence"])
    assert result.exit_code == 1 and "cannot ablate" in result.output


# --- GET /api/results ------------------------------------------------------------------


def _write_result(path: Path, date: str, preset: str) -> None:
    body = {
        "meta": {"set": "seeded", "preset": preset, "date": date, "ablate": []},
        "metrics": {"silent_failures": 0},
        "items": [{"id": "x"}],
    }
    path.write_text(json.dumps(body), encoding="utf-8")


def test_results_endpoint_lists_summaries_newest_first(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    _write_result(results / "2026-10-01-seeded-fast.json", "2026-10-01", "fast")
    _write_result(results / "2026-10-09-seeded-balanced.json", "2026-10-09", "balanced")
    (results / "broken.json").write_text("{nope", encoding="utf-8")
    (results / "list.json").write_text("[1, 2]", encoding="utf-8")
    (results / "notes.md").write_text("# not json", encoding="utf-8")
    nested = results / "2026-10-09-seeded-balanced" / "items"
    nested.mkdir(parents=True)
    _write_result(nested / "seeded-p01.json", "2030-01-01", "nested")  # never listed
    outside = tmp_path / "secret.json"
    _write_result(outside, "2099-01-01", "outside")

    app = create_app(lambda p: Guard(p), results_dir=results)
    client = TestClient(app)
    res = client.get("/api/results")
    assert res.status_code == 200
    body = res.json()
    assert [r["name"] for r in body] == ["2026-10-09-seeded-balanced", "2026-10-01-seeded-fast"]
    assert all(set(r) == {"name", "meta", "metrics"} for r in body)  # item lists not served
    assert client.get("/api/results/..%2Fsecret").status_code in (404, 405)


def test_results_endpoint_missing_dir_is_empty(tmp_path: Path) -> None:
    app = create_app(lambda p: Guard(p), results_dir=tmp_path / "none")
    assert TestClient(app).get("/api/results").json() == []
