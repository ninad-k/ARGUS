"""Settings defaults and nested environment-variable overrides."""

from argus.config import AppSettings


def test_defaults_load_without_env() -> None:
    settings = AppSettings(_env_file=None)  # type: ignore[call-arg]
    assert settings.llm.provider == "ollama"
    assert settings.llm.model == "gemma3:4b"
    assert settings.data.universe_size_per_market == 300
    assert settings.paper.max_positions == 10
    assert settings.scheduler.us_post_close == "16:30"
    assert settings.db_url.endswith("/argus.db")
    assert settings.db_url.startswith("sqlite+aiosqlite:///")


def test_nested_env_override(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("ARGUS_LLM__MODEL", "llama3:8b")
    monkeypatch.setenv("ARGUS_PAPER__MAX_POSITIONS", "25")
    settings = AppSettings(_env_file=None)  # type: ignore[call-arg]
    assert settings.llm.model == "llama3:8b"
    assert settings.paper.max_positions == 25
