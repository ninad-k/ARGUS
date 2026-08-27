"""Settings for the daily post-close screening scheduler."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class SchedulerSettings(BaseSettings):
    """When each market's post-close screen run fires, in that market's own timezone."""

    model_config = SettingsConfigDict(env_prefix="ARGUS_SCHEDULER_", extra="ignore")

    enabled: bool = True
    us_post_close: str = "16:30"
    us_timezone: str = "America/New_York"
    india_post_close: str = "18:30"
    india_timezone: str = "Asia/Kolkata"
