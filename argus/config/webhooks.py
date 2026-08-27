"""Settings for inbound webhook receivers (currently: TradingView alerts)."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class WebhookSettings(BaseSettings):
    """An empty ``tradingview_token`` disables the TradingView webhook endpoint
    entirely (every request 404s) -- ARGUS ships with no webhook receiver
    active by default."""

    model_config = SettingsConfigDict(env_prefix="ARGUS_WEBHOOKS_", extra="ignore")

    tradingview_token: str = ""
