"""Fan-out provider that tries an ordered list of price providers in turn."""

from datetime import UTC, date, datetime

import numpy as np
import structlog
from numpy.typing import NDArray

from argus.data.prices.base import (
    BAR_DTYPE,
    PriceDataProvider,
    ProviderHealth,
    Quote,
    aclose_if_closeable,
)
from argus.markets import Instrument, Market
from argus.markets.registry import get_market

logger = structlog.get_logger(__name__)


class CompositePriceProvider:
    """Tries each supporting provider in priority order, returns the first
    non-empty/non-``None`` result."""

    name = "composite"

    def __init__(self, providers: list[PriceDataProvider]) -> None:
        self._providers = providers

    def supports(self, market: Market) -> bool:
        return any(p.supports(market) for p in self._providers)

    async def get_daily_bars(
        self, inst: Instrument, start: date, end: date
    ) -> NDArray[np.void]:
        market = _market_for(inst)
        for provider in self._providers:
            if market is not None and not provider.supports(market):
                continue
            bars = await provider.get_daily_bars(inst, start, end)
            if len(bars) > 0:
                return bars
            logger.debug(
                "composite.get_daily_bars.empty",
                provider=provider.name,
                symbol=inst.symbol,
                market=inst.market_code,
            )
        return np.zeros(0, dtype=BAR_DTYPE)

    async def get_intraday_bars(
        self, inst: Instrument, interval: str = "15m", lookback_days: int = 30
    ) -> NDArray[np.void]:
        market = _market_for(inst)
        for provider in self._providers:
            if market is not None and not provider.supports(market):
                continue
            bars = await provider.get_intraday_bars(inst, interval, lookback_days)
            if len(bars) > 0:
                return bars
            logger.debug(
                "composite.get_intraday_bars.empty",
                provider=provider.name,
                symbol=inst.symbol,
                market=inst.market_code,
            )
        return np.zeros(0, dtype=BAR_DTYPE)

    async def get_quote(self, inst: Instrument) -> Quote | None:
        market = _market_for(inst)
        for provider in self._providers:
            if market is not None and not provider.supports(market):
                continue
            quote = await provider.get_quote(inst)
            if quote is not None:
                return quote
            logger.debug(
                "composite.get_quote.none",
                provider=provider.name,
                symbol=inst.symbol,
                market=inst.market_code,
            )
        return None

    async def health_check(self) -> ProviderHealth:
        results = [await p.health_check() for p in self._providers]
        ok = any(r.ok for r in results) if results else False
        pairs = zip(self._providers, results, strict=True)
        detail = "; ".join(f"{p.name}={r.detail}" for p, r in pairs) or "no providers configured"
        return ProviderHealth(ok=ok, detail=detail, checked_at=datetime.now(UTC))

    async def aclose(self) -> None:
        """Close every member provider that owns a closeable resource (e.g.
        ``NSEProvider``'s ``httpx.AsyncClient``). Members with nothing to
        close (most providers) are no-ops via ``aclose_if_closeable``."""
        for provider in self._providers:
            await aclose_if_closeable(provider)


def _market_for(inst: Instrument) -> Market | None:
    try:
        return get_market(inst.market_code)
    except KeyError:
        return None
