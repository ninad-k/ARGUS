"""Settings for the paper-trading simulator. No broker/execution code exists here — ever."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class PaperSettings(BaseSettings):
    """Simulated portfolio parameters, per market currency."""

    model_config = SettingsConfigDict(env_prefix="ARGUS_PAPER_", extra="ignore")

    starting_cash_us: float = 100_000.0
    starting_cash_india: float = 1_000_000.0
    slippage_bps: int = 5
    position_size_pct: float = 5.0
    max_positions: int = 10
