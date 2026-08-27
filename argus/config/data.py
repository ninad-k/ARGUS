"""Settings for market data providers and universe construction."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class DataSettings(BaseSettings):
    """Data-layer knobs: how much history to pull and how big a universe to scan."""

    model_config = SettingsConfigDict(env_prefix="ARGUS_DATA_", extra="ignore")

    universe_size_per_market: int = 300
    bar_lookback_days: int = 400
    provider_timeout_seconds: float = 30.0
    provider_max_retries: int = 3
