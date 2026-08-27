"""Settings for the local LLM used to narrate/rank daily picks."""

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    """LLM backend config. Defaults target a local Ollama server — no cloud key needed."""

    model_config = SettingsConfigDict(env_prefix="ARGUS_LLM_", extra="ignore")

    enabled: bool = True
    provider: str = "ollama"
    model: str = "gemma3:4b"
    base_url: str = "http://localhost:11434"
    api_key: SecretStr | None = None
    timeout_seconds: int = 120
