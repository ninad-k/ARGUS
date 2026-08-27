"""DB-backed registry of configured price-data sources.

``DataSource`` rows (see ``argus.db.models``) describe which providers are
enabled, in what priority order, and with what config. This module turns
those rows into live ``PriceDataProvider`` instances.
"""

from datetime import UTC, datetime
from typing import Any

import structlog
from sqlalchemy import select

from argus.config import AppSettings, get_settings
from argus.data.prices.base import PriceDataProvider, ProviderHealth, aclose_if_closeable
from argus.data.prices.composite import CompositePriceProvider
from argus.data.prices.nse_provider import NSEProvider
from argus.data.prices.static_provider import StaticPriceProvider
from argus.data.prices.tv_screener_provider import TVScreenerProvider
from argus.data.prices.yfinance_provider import YFinanceProvider
from argus.data.universe import SeedUniverseProvider, TVUniverseProvider, UniverseProvider
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
    if kind == "tvscreener":
        return TVScreenerProvider()
    if kind == "nse":
        return NSEProvider()
    logger.warning("sources.build_price_provider.unknown_kind", kind=kind)
    return YFinanceProvider()


async def resolve_universe_provider(
    market_code: str, settings: AppSettings | None = None
) -> UniverseProvider:
    """Build the pipeline's default ``UniverseProvider`` for ``market_code``.

    When an enabled ``tvscreener`` ``DataSource`` row exists *and is scoped to
    cover* ``market_code`` (its ``markets_json`` list is empty -- meaning all
    markets -- or contains ``market_code``), use a live TradingView top-liquid
    universe (``TVUniverseProvider``), falling back to seed CSVs on any
    failure/empty result. Otherwise, seeds only -- matching the pre-Task-9
    default so installs with no configured sources keep working exactly as
    before, and so an operator who scopes a tvscreener source to e.g. IN_NSE
    only doesn't silently get TV universes for US markets too.
    """
    resolved_settings = settings or get_settings()
    sources = await load_enabled_sources(resolved_settings)
    tv_source = next(
        (s for s in sources if s.kind == "tvscreener" and _source_covers_market(s, market_code)),
        None,
    )
    if tv_source is None:
        return SeedUniverseProvider()

    tv_provider = TVScreenerProvider()
    size = resolved_settings.data.universe_size_per_market
    return TVUniverseProvider(tv_provider, SeedUniverseProvider(), size)


def _source_covers_market(source: DataSource, market_code: str) -> bool:
    """Whether ``source`` is scoped to serve ``market_code``.

    An empty ``markets`` list means "all markets" (matches how sources are
    seeded by ``ensure_default_sources``/created with no explicit scoping).
    """
    markets = source.markets_json.get("markets", [])
    return not markets or market_code in markets


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


async def list_sources(settings: AppSettings | None = None) -> list[DataSource]:
    """Return every ``DataSource`` row (enabled or not), ordered by priority.

    Used by the sources admin UI/API, which needs to see disabled rows too
    (unlike ``load_enabled_sources``, used by the pipeline).
    """
    async with async_session(settings) as session:
        result = await session.execute(select(DataSource).order_by(DataSource.priority))
        return list(result.scalars().all())


async def create_source(
    name: str,
    kind: str,
    markets: list[str],
    config: dict[str, Any],
    priority: int = 0,
    settings: AppSettings | None = None,
) -> DataSource:
    """Create and persist a new ``DataSource`` row, enabled by default."""
    async with async_session(settings) as session:
        source = DataSource(
            name=name,
            kind=kind,
            markets_json={"markets": markets},
            config_json=config,
            priority=priority,
            enabled=True,
            last_health=None,
            created_at=datetime.now(UTC),
        )
        session.add(source)
        await session.commit()
        await session.refresh(source)
        return source


async def get_source(source_id: int, settings: AppSettings | None = None) -> DataSource | None:
    async with async_session(settings) as session:
        return await session.get(DataSource, source_id)


async def update_source(
    source_id: int,
    *,
    name: str | None = None,
    kind: str | None = None,
    markets: list[str] | None = None,
    config: dict[str, Any] | None = None,
    priority: int | None = None,
    enabled: bool | None = None,
    settings: AppSettings | None = None,
) -> DataSource | None:
    """Apply the given (non-``None``) field updates to a source. ``None`` -> not returned."""
    async with async_session(settings) as session:
        source = await session.get(DataSource, source_id)
        if source is None:
            return None
        if name is not None:
            source.name = name
        if kind is not None:
            source.kind = kind
        if markets is not None:
            source.markets_json = {"markets": markets}
        if config is not None:
            source.config_json = config
        if priority is not None:
            source.priority = priority
        if enabled is not None:
            source.enabled = enabled
        await session.commit()
        await session.refresh(source)
        return source


async def delete_source(source_id: int, settings: AppSettings | None = None) -> bool:
    """Delete a source by id. Returns whether a row was actually deleted."""
    async with async_session(settings) as session:
        source = await session.get(DataSource, source_id)
        if source is None:
            return False
        await session.delete(source)
        await session.commit()
        return True


async def check_source_health(
    source: DataSource, settings: AppSettings | None = None
) -> ProviderHealth:
    """Run the provider's health check and persist the result on ``source``.

    The provider built here is throwaway -- closed in ``finally`` (if it owns
    a closeable resource, e.g. ``NSEProvider``'s ``httpx.AsyncClient``) so a
    one-off health check never leaks a client.
    """
    provider = build_price_provider(source.kind, source.config_json)
    try:
        health = await provider.health_check()
    finally:
        await aclose_if_closeable(provider)

    async with async_session(settings) as session:
        db_source = await session.get(DataSource, source.id)
        if db_source is not None:
            db_source.last_health = health.detail if health.ok else f"UNHEALTHY: {health.detail}"
            await session.commit()

    return health
