"""Async engine/session management for the control-plane SQLite database."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from argus.config import AppSettings, get_settings
from argus.db.models import Base

# Keyed by db_url so tests using a distinct tmp_path AppSettings get their own
# engine/pool instead of colliding with the process-wide default.
_engines: dict[str, AsyncEngine] = {}


def get_engine(settings: AppSettings | None = None) -> AsyncEngine:
    settings = settings or get_settings()
    if settings.db_url not in _engines:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        _engines[settings.db_url] = create_async_engine(settings.db_url)
    return _engines[settings.db_url]


def get_sessionmaker(settings: AppSettings | None = None) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(get_engine(settings), expire_on_commit=False)


async def init_db(settings: AppSettings | None = None) -> None:
    """Create all tables. Idempotent — safe to call on every startup."""
    engine = get_engine(settings)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@asynccontextmanager
async def async_session(settings: AppSettings | None = None) -> AsyncIterator[AsyncSession]:
    sessionmaker = get_sessionmaker(settings)
    async with sessionmaker() as session:
        yield session
