"""Settings for the FastAPI + NiceGUI web server."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class UISettings(BaseSettings):
    """Where the ``argus`` console script binds the combined API/UI server."""

    model_config = SettingsConfigDict(env_prefix="ARGUS_UI_", extra="ignore")

    host: str = "127.0.0.1"
    port: int = 8321
