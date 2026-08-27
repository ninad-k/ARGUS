"""DB-backed registry of configured price-data sources.

``DataSource`` rows (see ``argus.db.models``) describe which providers are
enabled, in what priority order, and with what config. This module turns
those rows into live ``PriceDataProvider`` instances.
"""

from datetime import UTC, datetime
from typing import Any

import structlog
from sqlalchemy import select

from argus.config import AppSettings
from argus.data.prices.base import PriceDataProvider, ProviderHealth
from argus.data.prices.composite import CompositePriceProvider
from argus.data.prices.static_provider import StaticPriceProvider
from argus.data.prices.yfinance_provider import YFinanceProvider
from argus.db import async_session
from argus.db.models import DataSource

logger = structlog.get_logger(__name__)

_DEFAULT_YFINANCE_NAME = "yfinance-default"


async def load_enabled_sources(settings: AppSettings | None = None) -> list[DataSource]:
    """Return enabled ``DataSource`` rows, ordered by priority (ascending)."""
    async with async_session(settings) as session:
        result = await session.execute(
            select(DataSource).where(DataSource.enabled.is_(True)).order_by(DataSource.priority)
        )
        return list(result.scalars().all())


def build_price_provider(kind: str, config: dict[str, Any]) -> PriceDataProvider:
    """Construct a ``PriceDataProvider`` from a ``DataSource.kind`` + config dict.

    Falls back to ``YFinanceProvider`` for unknown kinds — never raises.
    """
    if kind == "yfinance":
        return YFinanceProvider()
    if kind == "static":
        return StaticPriceProvider()
    logger.warning("sources.build_price_provider.unknown_kind", kind=kind)
    return YFinanceProvider()


async def build_composite_from_db(settings: AppSettings | None = None) -> CompositePriceProvider:
    """Build a ``CompositePriceProvider`` from enabled ``DataSource`` rows.

    Falls back to a single default ``YFinanceProvider`` if no rows are
    configured (or the table is empty) so the pipeline works out of the box.
    """
    sources = await load_enabled_sources(settings)
    if not sources:
        logger.info("sources.build_composite_from_db.no_rows_falling_back_to_yfinance")
        return CompositePriceProvider([YFinanceProvider()])

    providers = [build_price_provider(source.kind, source.config_json) for source in sources]
    return CompositePriceProvider(providers)


async def ensure_default_sources(settings: AppSettings | None = None) -> None:
    """Seed a default enabled yfinance ``DataSource`` row if the table is empty."""
    async with async_session(settings) as session:
        result = await session.execute(select(DataSource))
        if result.scalars().first() is not None:
            return
        session.add(
            DataSource(
                name=_DEFAULT_YFINANCE_NAME,
                kind="yfinance",
                markets_json={"markets": ["US_NYSE", "US_NASDAQ", "IN_NSE"]},
                config_json={},
                priority=0,
                enabled=True,
                last_health=None,
                created_at=datetime.now(UTC),
            )
        )
        await session.commit()


async def check_source_health(
    source: DataSource, settings: AppSettings | None = None
) -> ProviderHealth:
    """Run the provider's health check and persist the result on ``source``."""
    provider = build_price_provider(source.kind, source.config_json)
    health = await provider.health_check()

    async with async_session(settings) as session:
        db_source = await session.get(DataSource, source.id)
        if db_source is not None:
            db_source.last_health = health.detail if health.ok else f"UNHEALTHY: {health.detail}"
            await session.commit()

    return health
