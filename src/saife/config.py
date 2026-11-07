"""Model routing and environment settings.

Presets, budgets and the full ``GuardConfig`` arrive in M1. M0 only needs to
know which model serves each pipeline role and how to reach the verifier.
"""

from __future__ import annotations

import json
import os
from enum import StrEnum
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

from saife.errors import ConfigError
from saife.redact import register_secrets

ANTHROPIC = "anthropic"
VERIFIER = "verifier"
OPENROUTER = "openrouter"
PROVIDERS = (ANTHROPIC, VERIFIER, OPENROUTER)
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# AI Lodge deployments read the club's names first (API_BASE_URL / API_KEY / MODEL_NAME) and fall
# back to the SAIFE_VERIFIER_* names, so both work. The base URL defaults to the club gateway.
DEFAULT_BASE_URL = "https://api.apiyi.com/v1"

StructuredMode = Literal["json_schema", "json_object", "text"]
STRUCTURED_MODES: tuple[StructuredMode, ...] = ("json_schema", "json_object", "text")

PROBE_PATH = Path(".saife/verifier_probe.json")

# "verifier" (default): every role runs on the verifier model; only the verifier key is
# needed and no Anthropic client is created. The gateway has no web search, so the
# web_evidence detector reports UNVERIFIABLE.
# "split": the original PRD routing (Claude + a second-provider verifier); needs
# ANTHROPIC_API_KEY.
Routing = Literal["split", "verifier"]
ROUTINGS: tuple[Routing, ...] = ("split", "verifier")


class Role(StrEnum):
    GENERATE = "generate"
    DECOMPOSE = "decompose"
    WEB_RETRIEVAL = "web_retrieval"
    GROUNDING_JUDGE = "grounding_judge"
    CONSISTENCY_JUDGE = "consistency_judge"
    WEB_JUDGE = "web_judge"
    REPAIR = "repair"


VERIFIER_ROLES = frozenset({Role.GROUNDING_JUDGE, Role.CONSISTENCY_JUDGE, Role.WEB_JUDGE})


class ModelRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    model: str


class FallbackPolicy(StrEnum):
    # Verifier unavailable -> use the fallback model and mark the run as lacking diversity.
    FALLBACK_AND_DISCLOSE = "fallback_and_disclose"
    # Verifier unavailable -> the error propagates and the claim becomes UNVERIFIABLE.
    FAIL_CLOSED = "fail_closed"


GENERATOR_MODEL = "claude-haiku-4-5"
ORCHESTRATOR_MODEL = "claude-opus-5"
DEFAULT_VERIFIER_MODEL = "gemini-3.1-flash-lite"


def default_roles(verifier_model: str = DEFAULT_VERIFIER_MODEL) -> dict[Role, ModelRef]:
    claude_main = ModelRef(provider=ANTHROPIC, model=ORCHESTRATOR_MODEL)
    verifier = ModelRef(provider=VERIFIER, model=verifier_model)
    return {
        Role.GENERATE: ModelRef(provider=ANTHROPIC, model=GENERATOR_MODEL),
        Role.DECOMPOSE: claude_main,
        Role.WEB_RETRIEVAL: claude_main,
        Role.GROUNDING_JUDGE: verifier,
        Role.CONSISTENCY_JUDGE: verifier,
        Role.WEB_JUDGE: verifier,
        Role.REPAIR: claude_main,
    }


def verifier_roles(verifier_model: str = DEFAULT_VERIFIER_MODEL) -> dict[Role, ModelRef]:
    """Every role on the verifier model. Web retrieval maps there too, but the verifier
    client cannot search, so web checks fail closed as UNVERIFIABLE (never silently skipped)."""
    return dict.fromkeys(Role, ModelRef(provider=VERIFIER, model=verifier_model))


VERIFIER_FALLBACK = ModelRef(provider=ANTHROPIC, model=ORCHESTRATOR_MODEL)


class Settings(BaseModel):
    """Values read from the environment (and ``.env``). Keys are never logged."""

    anthropic_api_key: str | None = None
    verifier_api_key: str | None = None
    verifier_base_url: str | None = None
    verifier_model: str = DEFAULT_VERIFIER_MODEL
    verifier_structured_mode: StructuredMode = "json_schema"
    routing: Routing = "verifier"
    openrouter_api_key: str | None = None
    openrouter_base_url: str = OPENROUTER_BASE_URL
    # Per-role overrides from SAIFE_ROLE_<ROLE>=provider:model, applied on top of routing.
    role_overrides: dict[Role, ModelRef] = Field(default_factory=dict)
    # The user's own sAIfe Monitor deployment (dashboard/), optional.
    dashboard_url: str | None = None
    dashboard_secret: str | None = None

    @property
    def dashboard_host(self) -> str | None:
        if not self.dashboard_url:
            return None
        return self.dashboard_url.split("//", 1)[-1].split("/", 1)[0]

    def __repr__(self) -> str:  # never show keys
        host = self.verifier_host or "-"
        mode = self.verifier_structured_mode
        return (
            f"Settings(verifier={host}/{self.verifier_model}, mode={mode}, routing={self.routing})"
        )

    __str__ = __repr__

    def roles(self) -> dict[Role, ModelRef]:
        base = (
            verifier_roles(self.verifier_model)
            if self.routing == "verifier"
            else default_roles(self.verifier_model)
        )
        return {**base, **self.role_overrides}

    def require_openrouter(self) -> str:
        if not self.openrouter_api_key:
            raise ConfigError(
                "A role is routed to OpenRouter but OPENROUTER_API_KEY is not set in .env."
            )
        return self.openrouter_api_key

    @property
    def verifier_host(self) -> str | None:
        if not self.verifier_base_url:
            return None
        return self.verifier_base_url.split("//", 1)[-1].split("/", 1)[0]

    def require_verifier(self) -> tuple[str, str]:
        if not self.verifier_api_key or not self.verifier_base_url:
            raise ConfigError(
                "Verifier not configured: set API_KEY (or SAIFE_VERIFIER_API_KEY) and "
                "API_BASE_URL (or SAIFE_VERIFIER_BASE_URL) in .env (see .env.example)."
            )
        return self.verifier_api_key, self.verifier_base_url


def _probe_mode(path: Path) -> StructuredMode | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    mode = data.get("best_mode")
    return mode if mode in STRUCTURED_MODES else None


_LOCAL_HOSTS = ("localhost", "127.0.0.1", "[::1]")


def _dashboard() -> tuple[str | None, str | None]:
    """``SAIFE_DASHBOARD_URL`` + ``SAIFE_DASHBOARD_SECRET``: both or neither. The URL must
    be https (plain http only for a local dev server), since runs carry claims and sources."""
    url = (os.getenv("SAIFE_DASHBOARD_URL") or "").strip().rstrip("/") or None
    secret = os.getenv("SAIFE_DASHBOARD_SECRET") or None
    if not url and not secret:
        return None, None
    if not url or not secret:
        raise ConfigError("set both SAIFE_DASHBOARD_URL and SAIFE_DASHBOARD_SECRET, or neither")
    host = url.split("//", 1)[-1].split("/", 1)[0].rsplit(":", 1)[0]
    if not (url.startswith("https://") or (url.startswith("http://") and host in _LOCAL_HOSTS)):
        raise ConfigError(
            "SAIFE_DASHBOARD_URL must start with https:// (http:// only for localhost)"
        )
    if len(secret) < 16:
        raise ConfigError("SAIFE_DASHBOARD_SECRET must be at least 16 characters")
    return url, secret


def _role_overrides() -> dict[Role, ModelRef]:
    """``SAIFE_ROLE_<ROLE>=provider:model``, e.g.
    ``SAIFE_ROLE_GENERATE=openrouter:nvidia/nemotron-3-super-120b-a12b:free``.
    Only the first ``:`` splits, because model ids may contain one."""
    out: dict[Role, ModelRef] = {}
    for role in Role:
        name = f"SAIFE_ROLE_{role.name}"
        raw = (os.getenv(name) or "").strip()
        if not raw:
            continue
        provider, sep, model = raw.partition(":")
        if not sep or provider not in PROVIDERS or not model:
            raise ConfigError(f"{name} must look like provider:model with provider in {PROVIDERS}")
        out[role] = ModelRef(provider=provider, model=model)
    # GENERATE_MODEL_NAME: the subject model on the same endpoint, from another lab than the
    # judges (P3). An explicit SAIFE_ROLE_GENERATE wins.
    generate = (os.getenv("GENERATE_MODEL_NAME") or "").strip()
    if generate and Role.GENERATE not in out:
        out[Role.GENERATE] = ModelRef(provider=VERIFIER, model=generate)
    return out


def _env(*names: str) -> str | None:
    """The first non-empty value among ``names``."""
    for name in names:
        value = (os.getenv(name) or "").strip()
        if value:
            return value
    return None


def load_settings(*, env_file: str | None = ".env", probe_path: Path = PROBE_PATH) -> Settings:
    """Load settings. Mode precedence: env override > probe result > json_schema."""
    if env_file:
        load_dotenv(env_file, override=False)

    mode_env = os.getenv("SAIFE_VERIFIER_STRUCTURED_MODE") or None
    if mode_env is not None and mode_env not in STRUCTURED_MODES:
        raise ConfigError(f"SAIFE_VERIFIER_STRUCTURED_MODE must be one of {STRUCTURED_MODES}")
    mode = mode_env or _probe_mode(probe_path) or "json_schema"

    routing = os.getenv("SAIFE_ROUTING") or "verifier"
    if routing not in ROUTINGS:
        raise ConfigError(f"SAIFE_ROUTING must be one of {ROUTINGS}")

    dashboard_url, dashboard_secret = _dashboard()
    settings = Settings(
        dashboard_url=dashboard_url,
        dashboard_secret=dashboard_secret,
        role_overrides=_role_overrides(),
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY") or os.getenv("API_KEY") or None,
        openrouter_base_url=os.getenv("OPENROUTER_BASE_URL") or OPENROUTER_BASE_URL,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
        verifier_api_key=_env("API_KEY", "SAIFE_VERIFIER_API_KEY"),
        verifier_base_url=_env("API_BASE_URL", "SAIFE_VERIFIER_BASE_URL") or DEFAULT_BASE_URL,
        verifier_model=_env("MODEL_NAME", "SAIFE_VERIFIER_MODEL") or DEFAULT_VERIFIER_MODEL,
        verifier_structured_mode=mode,
        routing=routing,
    )
    register_secrets(
        [
            settings.anthropic_api_key,
            settings.verifier_api_key,
            settings.openrouter_api_key,
            settings.dashboard_secret,
        ]
    )
    return settings


# --- Presets and budgets (PRD §4.4) ------------------------------------------------

DetectorKey = Literal["grounding", "consistency", "web_evidence", "deterministic"]


class Budgets(BaseModel):
    """Hard caps on a run (P4). Hitting any of them stops the loop; remaining
    unverified claims become UNVERIFIABLE and the answer is at least CAUTION."""

    model_config = ConfigDict(frozen=True)

    max_passes: int = 2
    max_tokens: int = 400_000  # input + output, all calls in the run
    timeout_s: float = 120.0


class GuardConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    preset: str
    detectors: tuple[DetectorKey, ...]
    budgets: Budgets
    repair: bool = True  # fast has no repair
    consistency_samples: int = 3
    web_max_uses: int = 3
    # Claims with these statuses are repaired (hedged/removed/corrected).
    repair_statuses: tuple[str, ...] = ("CONTRADICTED", "UNSUPPORTED")
    block_on_contradicted: bool = False  # strict: any surviving CONTRADICTED → BLOCKED
    block_risk_threshold: float = 0.6  # answer risk above this → BLOCKED
    fallback_policy: FallbackPolicy = FallbackPolicy.FALLBACK_AND_DISCLOSE
    roles: dict[Role, ModelRef] | None = None  # None → default_roles(settings.verifier_model)

    def with_overrides(self, **overrides: object) -> GuardConfig:
        """Per-run overrides (F12). ``budgets`` may be passed as a dict of fields."""
        budgets = overrides.pop("budgets", None)
        if isinstance(budgets, dict):
            # model_copy(update=) skips validation; re-validate so bad overrides are rejected.
            overrides["budgets"] = Budgets.model_validate({**self.budgets.model_dump(), **budgets})
        elif budgets is not None:
            overrides["budgets"] = budgets
        return self.model_validate({**self.model_dump(), **overrides})


PRESETS: dict[str, GuardConfig] = {
    "fast": GuardConfig(
        preset="fast",
        detectors=("grounding", "deterministic"),
        budgets=Budgets(max_passes=1, max_tokens=150_000, timeout_s=45.0),
        repair=False,
        consistency_samples=0,
    ),
    "balanced": GuardConfig(
        preset="balanced",
        detectors=("grounding", "consistency", "deterministic"),
        budgets=Budgets(max_passes=2, max_tokens=400_000, timeout_s=120.0),
        consistency_samples=3,
    ),
    "strict": GuardConfig(
        preset="strict",
        detectors=("grounding", "consistency", "web_evidence", "deterministic"),
        budgets=Budgets(max_passes=3, max_tokens=800_000, timeout_s=240.0),
        consistency_samples=5,
        block_on_contradicted=True,
        block_risk_threshold=0.4,
        fallback_policy=FallbackPolicy.FAIL_CLOSED,
    ),
}


def get_preset(name: str) -> GuardConfig:
    try:
        return PRESETS[name]
    except KeyError:
        raise ConfigError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}") from None
