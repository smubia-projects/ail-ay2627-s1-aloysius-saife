from __future__ import annotations

import json
from pathlib import Path

import pytest

from saife.config import VERIFIER_ROLES, ConfigError, Role, default_roles, load_settings


def test_default_routing_matches_prd() -> None:
    roles = default_roles()
    assert roles[Role.GENERATE].model == "claude-haiku-4-5"
    assert roles[Role.DECOMPOSE].model == "claude-opus-5"
    assert roles[Role.REPAIR].model == "claude-opus-5"
    for role in VERIFIER_ROLES:
        assert roles[role].provider == "verifier"
        assert roles[role].model == "gemini-3.1-flash-lite"


def test_mode_precedence_env_over_probe(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    probe = tmp_path / "probe.json"
    probe.write_text(json.dumps({"best_mode": "text"}))
    assert load_settings(env_file=None, probe_path=probe).verifier_structured_mode == "text"

    monkeypatch.setenv("SAIFE_VERIFIER_STRUCTURED_MODE", "json_object")
    assert load_settings(env_file=None, probe_path=probe).verifier_structured_mode == "json_object"


def test_default_mode_without_probe(tmp_path: Path) -> None:
    s = load_settings(env_file=None, probe_path=tmp_path / "missing.json")
    assert s.verifier_structured_mode == "json_schema"


def test_invalid_mode_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SAIFE_VERIFIER_STRUCTURED_MODE", "yaml")
    with pytest.raises(ConfigError):
        load_settings(env_file=None, probe_path=tmp_path / "x.json")


def test_require_verifier_without_key(tmp_path: Path) -> None:
    s = load_settings(env_file=None, probe_path=tmp_path / "x.json")
    with pytest.raises(ConfigError, match="SAIFE_VERIFIER_API_KEY"):
        s.require_verifier()


def test_overrides_validate_budgets() -> None:
    from pydantic import ValidationError

    from saife.config import get_preset

    cfg = get_preset("balanced").with_overrides(budgets={"max_passes": 5})
    assert cfg.budgets.max_passes == 5 and cfg.budgets.timeout_s == 120.0
    with pytest.raises(ValidationError):
        get_preset("balanced").with_overrides(budgets={"max_passes": "many"})
    with pytest.raises(ConfigError):
        get_preset("yolo")


def test_routing_defaults_to_verifier_only(tmp_path: Path) -> None:
    s = load_settings(env_file=None, probe_path=tmp_path / "x.json")
    assert s.routing == "verifier"
    assert {ref.provider for ref in s.roles().values()} == {"verifier"}


def test_split_routing_is_the_prd_routing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SAIFE_ROUTING", "split")
    assert load_settings(env_file=None, probe_path=tmp_path / "x.json").roles() == default_roles()


def test_verifier_routing_puts_every_role_on_the_verifier(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SAIFE_VERIFIER_MODEL", "gemini-x")
    roles = load_settings(env_file=None, probe_path=tmp_path / "x.json").roles()
    for role in Role:
        assert (roles[role].provider, roles[role].model) == ("verifier", "gemini-x")


def test_invalid_routing_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SAIFE_ROUTING", "gemini")
    with pytest.raises(ConfigError, match="SAIFE_ROUTING"):
        load_settings(env_file=None, probe_path=tmp_path / "x.json")


def test_verifier_routing_never_reaches_claude(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Gemini-only routing builds no Anthropic client, even when a Claude key is set."""
    from saife.config import FallbackPolicy
    from saife.llm import build_router
    from tests.conftest import FAKE_ANTHROPIC_KEY, FAKE_VERIFIER_KEY

    monkeypatch.setenv("SAIFE_VERIFIER_API_KEY", FAKE_VERIFIER_KEY)
    monkeypatch.setenv("SAIFE_VERIFIER_BASE_URL", "https://gw.example/v1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_ANTHROPIC_KEY)
    settings = load_settings(env_file=None, probe_path=tmp_path / "x.json")
    r = build_router(settings, policy=FallbackPolicy.FALLBACK_AND_DISCLOSE)
    assert set(r.clients) == {"verifier"}
    assert r.fallback_ref == r.roles[Role.GROUNDING_JUDGE]  # self-fallback = none
    assert not r.can_search(Role.WEB_RETRIEVAL)

    monkeypatch.setenv("SAIFE_ROUTING", "split")
    split = build_router(
        load_settings(env_file=None, probe_path=tmp_path / "x.json"),
        policy=FallbackPolicy.FALLBACK_AND_DISCLOSE,
    )
    assert set(split.clients) == {"anthropic", "verifier"}
    assert split.fallback_ref.provider == "anthropic"


def test_role_override_keeps_colons_in_model_ids(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SAIFE_ROLE_GENERATE", "openrouter:nvidia/nemotron-3-super-120b-a12b:free")
    roles = load_settings(env_file=None, probe_path=tmp_path / "x.json").roles()
    assert (roles[Role.GENERATE].provider, roles[Role.GENERATE].model) == (
        "openrouter",
        "nvidia/nemotron-3-super-120b-a12b:free",
    )
    assert roles[Role.GROUNDING_JUDGE].provider == "verifier"  # untouched roles keep routing


@pytest.mark.parametrize("raw", ["nemotron", "openrouter:", "acme:some-model", ":model"])
def test_bad_role_override_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, raw: str
) -> None:
    monkeypatch.setenv("SAIFE_ROLE_REPAIR", raw)
    with pytest.raises(ConfigError, match="SAIFE_ROLE_REPAIR"):
        load_settings(env_file=None, probe_path=tmp_path / "x.json")


def _gateway_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.conftest import FAKE_VERIFIER_KEY

    monkeypatch.setenv("SAIFE_VERIFIER_API_KEY", FAKE_VERIFIER_KEY)
    monkeypatch.setenv("SAIFE_VERIFIER_BASE_URL", "https://gw.example/v1")


def test_openrouter_client_only_when_a_role_uses_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from saife.llm import build_router

    _gateway_env(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-0123456789abcdef")
    r = build_router(load_settings(env_file=None, probe_path=tmp_path / "x.json"))
    assert set(r.clients) == {"verifier"}  # key alone does not add a client

    monkeypatch.setenv("SAIFE_ROLE_GENERATE", "openrouter:vendor/model:free")
    r = build_router(load_settings(env_file=None, probe_path=tmp_path / "x.json"))
    assert set(r.clients) == {"verifier", "openrouter"}
    assert r.clients["openrouter"].provider == "openrouter"
    assert r.roles[Role.GENERATE].provider == "openrouter"
    assert r.fallback_ref == r.roles[Role.GROUNDING_JUDGE]  # still no fallback to Claude


def test_openrouter_role_without_key_is_a_config_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from saife.llm import build_router

    _gateway_env(monkeypatch)
    monkeypatch.setenv("SAIFE_ROLE_GENERATE", "openrouter:vendor/model")
    with pytest.raises(ConfigError, match="OPENROUTER_API_KEY"):
        build_router(load_settings(env_file=None, probe_path=tmp_path / "x.json"))


def test_openrouter_key_is_redacted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from saife.redact import redact

    key = "sk-or-v1-" + "a1b2c3d4" * 8
    monkeypatch.setenv("OPENROUTER_API_KEY", key)
    s = load_settings(env_file=None, probe_path=tmp_path / "x.json")
    assert key not in repr(s) and key not in redact(f"auth {key}")


def test_club_env_names_win_over_saife_names(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for name in (
        "API_KEY",
        "API_BASE_URL",
        "MODEL_NAME",
        "GENERATE_MODEL_NAME",
        "SAIFE_ROLE_GENERATE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SAIFE_VERIFIER_API_KEY", "sk-old-0123456789abcdef")
    monkeypatch.setenv("SAIFE_VERIFIER_MODEL", "old-model")
    s = load_settings(env_file=None, probe_path=tmp_path / "x.json")
    assert s.verifier_base_url == "https://api.apiyi.com/v1"  # club gateway by default
    assert s.verifier_model == "old-model" and s.verifier_api_key == "sk-old-0123456789abcdef"

    monkeypatch.setenv("API_KEY", "sk-new-0123456789abcdef")
    monkeypatch.setenv("API_BASE_URL", "https://gw.example/v1")
    monkeypatch.setenv("MODEL_NAME", "gemini-3.1-flash-lite")
    monkeypatch.setenv("GENERATE_MODEL_NAME", "deepseek-v4-flash")
    s = load_settings(env_file=None, probe_path=tmp_path / "x.json")
    assert (s.verifier_api_key, s.verifier_base_url) == (
        "sk-new-0123456789abcdef",
        "https://gw.example/v1",
    )
    roles = s.roles()
    assert roles[Role.GENERATE].model == "deepseek-v4-flash"
    assert roles[Role.GROUNDING_JUDGE].model == "gemini-3.1-flash-lite"


def test_explicit_role_override_beats_generate_model_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("GENERATE_MODEL_NAME", "deepseek-v4-flash")
    monkeypatch.setenv("SAIFE_ROLE_GENERATE", "verifier:qwen3.6-flash")
    roles = load_settings(env_file=None, probe_path=tmp_path / "x.json").roles()
    assert roles[Role.GENERATE].model == "qwen3.6-flash"
