"""Settings for options analytics."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class OptionsSettings(BaseSettings):
    """Options-layer knobs. ``risk_free_rate`` feeds every Black-Scholes call
    that doesn't get an explicit ``r`` -- see ``argus.options.black_scholes``."""

    model_config = SettingsConfigDict(env_prefix="ARGUS_OPTIONS_", extra="ignore")

    risk_free_rate: float = 0.05
