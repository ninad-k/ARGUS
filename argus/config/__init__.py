"""Per-domain application settings."""

from argus.config.base import AppSettings, get_settings
from argus.config.data import DataSettings
from argus.config.llm import LLMSettings
from argus.config.paper import PaperSettings
from argus.config.scheduler import SchedulerSettings

__all__ = [
    "AppSettings",
    "DataSettings",
    "LLMSettings",
    "PaperSettings",
    "SchedulerSettings",
    "get_settings",
]
