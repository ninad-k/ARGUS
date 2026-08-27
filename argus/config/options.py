"""Settings for options analytics and the derivative suggester."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class OptionsSettings(BaseSettings):
    """Options-layer knobs. ``risk_free_rate`` feeds every Black-Scholes call
    that doesn't get an explicit ``r`` -- see ``argus.options.black_scholes``.

    The remaining fields configure ``argus.options.suggester`` (Task 12):
    ``enabled`` gates whether ``run_daily_pipeline`` attaches derivative
    suggestions to picks at all; ``risk_level`` is the default
    ``RiskLevel`` used when the pipeline builds suggestions (a plain
    ``str`` here, not the enum, to keep this a dependency-free settings
    module -- ``argus.options.suggester`` parses it); ``min_oi``/
    ``max_spread_pct`` are the liquidity filter; ``ivr_expensive_threshold``
    is the IV-Rank cutoff for the "expensive premium" guard; ``expiry_min_days``/
    ``expiry_max_days`` bound ``select_expiry``'s preferred window.
    """

    model_config = SettingsConfigDict(env_prefix="ARGUS_OPTIONS_", extra="ignore")

    risk_free_rate: float = 0.05

    enabled: bool = True
    risk_level: str = "moderate"
    min_oi: int = 100
    max_spread_pct: float = 10.0
    ivr_expensive_threshold: float = 70.0
    expiry_min_days: int = 20
    expiry_max_days: int = 60
