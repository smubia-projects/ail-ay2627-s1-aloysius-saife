# sAIfe

A multi-pass hallucination guard for agentic AI. Every answer is split into atomic claims, each claim is checked by independent detectors over a bounded number of passes, unsupported claims are repaired or removed, and the user is told plainly what was verified, what was changed and what could not be checked.

> Status: M0–M6 built; 901 offline tests pass; live-checked on the user's own keys. The full benchmark run is still pending. See `.claude/prds/saife.prd.md` (milestones) and `.claude/plans/saife.plan.md`.

## AI Lodge showcase deployment

This fork runs as a public demo on the SMU BIA AI Lodge stack:

- **Deployed:** the FastAPI server (`src/saife/server/`) on Google Cloud Run, built from the
  `Dockerfile` (only `src/` and the trap questions and benchmark summaries in `bench/` go into the
  image), and the demo UI (`web/`) on Vercel, which calls the API at `VITE_API_BASE`.
- **Not deployed (here in the repo):** the CLI, the Python SDK, the benchmark runner and the
  sAIfe Monitor dashboard (`dashboard/`). Run them on your own keys as described below.
- **One request per run.** `POST /api/runs/stream` runs the guard inside the request and streams its
  events (SSE); the last event carries the report. Nothing is stored on the server.
- **Presets:** `fast` and `balanced`. `strict` needs Claude web search and is not offered in the demo.
- **Models (club gateway, from env):** `MODEL_NAME=gemini-3.1-flash-lite` judges, splits and
  repairs; `GENERATE_MODEL_NAME=deepseek-v4-1-flash-260910` writes answers, so a different lab's model is
  checked (P3). The judges stay on Gemini: it is the model the student measured, and moving them to
  DeepSeek too would put the generator and its checkers in the same lab. `SAIFE_VERIFIER_STRUCTURED_MODE=json_object`, because the gateway accepts
  `json_schema` without enforcing it.
- **Limits:** each visitor gets a few checks (one per run, `RATE_LIMIT_CHECK_MAX` /
  `RATE_LIMIT_ASK_MAX`, tuned centrally). After that the API answers 429 and the page shows the
  AI Lodge invitation; no recorded run is substituted.
- Endpoints: `GET /api/health`, `POST /api/runs/stream`, `GET /api/presets`, `GET /api/traps`,
  `GET /api/results`. CORS comes from `CORS_ORIGINS`.

## Quickstart

sAIfe runs on **your own API keys**. The only required key is for an OpenAI-compatible verifier
endpoint; by default that is Google's own Gemini endpoint, with a free key from
[Google AI Studio](https://aistudio.google.com/apikey).

```bash
uv sync
cp .env.example .env                       # then put your key in SAIFE_VERIFIER_API_KEY
uv run python scripts/probe_verifier.py    # checks your key and picks the structured-output mode
uv run python -m saife config              # shows which model serves each role (never prints keys)
uv run pytest -q -m "not live"             # offline, free, deterministic; no keys needed
uv run python -m saife check -a examples/answer_with_errors.md -s examples/sources
```

## Keys and routing

Keys are read only from `.env` (gitignored) or the environment. They are redacted from logs and
never written to traces or reports. Choose the setup that matches the keys you have:

| Setup | Keys in `.env` | What runs where | Trade-off |
| --- | --- | --- | --- |
| **Minimum** | `SAIFE_VERIFIER_API_KEY` | Every role on the verifier model (Gemini by default) | `saife run` answers are judged by the model family that wrote them; no web evidence |
| **Recommended, free** | + `OPENROUTER_API_KEY`, `SAIFE_ROLE_GENERATE=openrouter:nvidia/nemotron-3-super-120b-a12b:free` | A different lab writes answers; Gemini judges, splits and repairs | Independent checking (P3) at no cost; free OpenRouter models allow 50 requests a day (1,000 after $10 of credit) |
| **Paid OpenRouter** | + `SAIFE_ROLE_GENERATE`, `_DECOMPOSE`, `_REPAIR` on `openrouter:anthropic/...` | Claude writes, splits and repairs; Gemini judges | Better repairs; needs OpenRouter credit; still no web evidence |
| **Original PRD** | + `ANTHROPIC_API_KEY`, `SAIFE_ROUTING=split` | Claude Haiku writes; Claude Opus splits, repairs and searches the web; Gemini judges | Adds strict's web evidence; Claude only as a disclosed judge fallback |

Any role can be moved with `SAIFE_ROLE_<ROLE>=provider:model`, where provider is `verifier`,
`openrouter` or `anthropic` (see `.env.example`). The verifier can be any OpenAI-compatible
endpoint: Google, OpenRouter (`https://openrouter.ai/api/v1`, model
`google/gemini-3.1-flash-lite`) or another gateway. Re-run `scripts/probe_verifier.py` after
changing it: some gateways accept `response_format: json_schema` without enforcing it, and the probe
picks the mode that actually returns valid output.

Without an Anthropic key, the strict preset's web checks come back UNVERIFIABLE, never silently
skipped. Claims and source excerpts go to the endpoints you configure; for sensitive data, use the
model provider's own endpoint rather than a third-party gateway.

Python SDK:

```python
import asyncio
from saife import Guard, Source


async def main() -> None:
    guard = Guard("balanced")
    report = await guard.check(
        "The Eiffel Tower was completed in 1889 and is 330 metres tall.",
        sources=[Source(id="eiffel", text=open("examples/sources/eiffel.md").read())],
    )
    print(report.status, report.final_answer)
    for c in report.claims:
        print(c.status, c.claim.text)


asyncio.run(main())
```

`guard.run(question)` generates the answer with the subject model first; `@guard.wrap` turns any async answer function into one that returns a checked `Report`.

CLI (use `python -m saife` where Windows App Control blocks the `saife.exe` shim):

```bash
uv run python -m saife check --answer examples/answer_with_errors.md --sources examples/sources --preset balanced
uv run python -m saife run "When was the Eiffel Tower built?" --preset strict
uv run python -m saife replay runs/<run_id>.jsonl      # rebuild a report from its trace, no model calls
uv run python -m saife bench --set seeded --preset balanced --limit 5
```

Server and demo UI:

```bash
uv run uvicorn saife.server.app:app --port 8000        # API only (the UI is web/)
cd web && npm ci && npm run dev                          # http://localhost:5173 (?demo=1 works without the API)
```

## Monitoring dashboard (optional)

`dashboard/` is **sAIfe Monitor**, a password-protected dashboard you deploy on Vercel's free
tier with a free Upstash Redis database. It shows which claims the judges flagged as
hallucinations and which passed, every repair, live progress of running checks, and
benchmark results. Each user deploys their own copy. Set `SAIFE_DASHBOARD_URL` and
`SAIFE_DASHBOARD_SECRET` in `.env`, and every run sends its events there (HMAC-signed);
`saife bench` publishes its summary, and `saife publish-bench <result.json>` sends an older one.
Setup takes about 10 minutes: see [dashboard/README.md](dashboard/README.md).

## How it works

```
 query ─▶ GENERATE ─▶ DECOMPOSE ─▶ VERIFY (detectors in parallel) ─▶ AGGREGATE ─▶ resolved? ─▶ DELIVER
                          ▲                                                       │ no
                          └──── re-verify new claims only ◀── REPAIR ◀────────────┘ (budget left, converging)
 every step ─▶ event bus ─▶ SSE (UI) · webhook (HMAC-signed) · console · callback · JSONL trace
```

| Detector | Catches | How the evidence is checked |
|---|---|---|
| Grounding | Claims unsupported by or contradicting the supplied sources | Judge returns exact quotes; code confirms each one is in the source |
| Self-consistency | Facts the subject model states differently on each resample | Agreement ratio across N samples; can only reach `LIKELY`, never `SUPPORTED` |
| Web evidence | World-knowledge errors when no sources are given | Claude web search, then a judge; quotes checked against the cited result text |
| Deterministic | Wrong arithmetic, impossible dates, dead URLs/DOIs, made-up stdlib APIs | Pure code, no LLM |

Claim status: `SUPPORTED` (verified evidence) · `CONTRADICTED` (verified counter-evidence) · `LIKELY` (consistency only) · `UNSUPPORTED` (checked, nothing found) · `UNVERIFIABLE` (check failed or claim not factual). Answer status: `VERIFIED`, `CAUTION` or `BLOCKED`.

Presets: `fast` (grounding + deterministic, 1 pass, no repair) · `balanced` (+ consistency N=3, 2 passes) · `strict` (+ web evidence, N=5, 3 passes, blocks on any contradiction, verifier never falls back).

## Safety principles mapped to code and tests

| # | Principle | Code | Tests |
|---|---|---|---|
| P1 | Fail closed | `errors.py`, `detectors/base.py` (`unverifiable`), `pipeline/loop.py` (`_run_detector`, `_finish`), `pipeline/aggregate.py` | `tests/invariants/`, `test_aggregate.py`, `test_loop.py` |
| P2 | Evidence checked by code | `evidence.py` (the only place `verified=True` is set), `detectors/base.py` (`enforce_evidence`) | `test_evidence.py`, `test_grounding.py`, `test_web_evidence.py`, `tests/invariants/` |
| P3 | Independent verifiers | `config.Settings.roles()` (generator and judges on different labs' models when a second provider is set), one-claim prompts in `detectors/grounding.py`, `llm/router.py` (fallback only on outages, disclosed) | `test_router.py`, `test_grounding.py` |
| P4 | Bounded loops | `config.Budgets`, `pipeline/loop.py` (`_passes`: passes, tokens, timeout, convergence) | `test_loop.py`, `tests/invariants/` |
| P5 | No silent edits | `pipeline/repair.py` (`RepairRecord` per edit), `trace.verify_trace` | `test_repair.py`, `tests/invariants/` |
| P6 | Abstention is valid | `pipeline/aggregate.answer_status`, `pipeline/loop.abstention` | `test_aggregate.py`, `test_loop.py` |
| P7 | Untrusted input is data | `prompts.data_block`, schema-validated outputs, `evidence.instruction_regions` | `tests/redteam/` |
| P8 | Full audit trail | `events.py`, `trace.py` (`TraceSink`, `replay`) | `test_trace.py`, `tests/integration/` |

Secrets come only from `.env` (gitignored); logs pass through `redact.RedactingFilter`, and traces store the provider host and model, never keys or headers (`test_secrets.py`).

## Model roles

Source of truth: `saife.config.Settings.roles()` (routing plus overrides). `uv run python -m saife config` prints the result.

| Role | Default (`verifier`) | `split` (needs `ANTHROPIC_API_KEY`) |
|---|---|---|
| Generate answers (subject) | verifier model, or `SAIFE_ROLE_GENERATE` | `claude-haiku-4-5` |
| Split claims, repair | verifier model, or overrides | `claude-opus-5` |
| Web retrieval (strict) | unavailable: claims UNVERIFIABLE | `claude-opus-5` web search |
| Verifier judgements | verifier model (default `gemini-3.1-flash-lite`) | verifier model; Claude only as a disclosed outage fallback |

## Limits

sAIfe reduces and discloses hallucination risk; it does not certify truth. A quote check proves a passage is in the source, not that the source is right. English only.
