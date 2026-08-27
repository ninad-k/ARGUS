"""Root application settings, composed from per-domain settings sub-models.

Every component reads settings via ``get_settings()`` — do not read
``os.environ`` anywhere else in the codebase.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from argus.config.data import DataSettings
from argus.config.llm import LLMSettings
from argus.config.paper import PaperSettings
from argus.config.scheduler import SchedulerSettings
from argus.config.ui import UISettings
from argus.config.webhooks import WebhookSettings


class AppSettings(BaseSettings):
    """Top-level settings. Nested fields are overridable via ``ARGUS_<DOMAIN>__<FIELD>``."""

    model_config = SettingsConfigDict(
        env_prefix="ARGUS_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path = Field(default_factory=lambda: Path.home() / ".argus")

    data: DataSettings = Field(default_factory=DataSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    scheduler: SchedulerSettings = Field(default_factory=SchedulerSettings)
    paper: PaperSettings = Field(default_factory=PaperSettings)
    ui: UISettings = Field(default_factory=UISettings)
    webhooks: WebhookSettings = Field(default_factory=WebhookSettings)

    @property
    def db_url(self) -> str:
        return f"sqlite+aiosqlite:///{self.data_dir}/argus.db"

    @property
    def duckdb_path(self) -> Path:
        return self.data_dir / "market_data.duckdb"


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return cached settings. Override via ``ARGUS_*`` environment variables."""
    return AppSettings()
