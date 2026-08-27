"""TradingView-screener-backed price provider.

The TradingView scanner is a point-in-time snapshot API, not an OHLCV history
endpoint -- ``get_daily_bars`` always returns empty here (documented, not a
bug; the composite provider falls through to another source, e.g. yfinance,
for history). This provider's value is bulk quotes (``get_quote``) and, via
the optional ``UniverseSource`` capability, a liquidity-ranked universe
(``top_liquid``) built from a single scanner query per market sorted by
dollar volume traded.

Column names verified empirically against the installed ``tradingview-screener``
3.2.1 (see ``argus.data.fundamentals.tv_fundamentals`` module docstring for
how -- unknown columns silently return ``None`` rather than raising):
``name`` (ticker), ``description`` (company name), ``sector``, ``close``,
``change`` (percent change from prior close), ``Value.Traded`` (dollar/rupee
volume traded), ``market_cap_basic``.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime
from typing import Any, Protocol, cast

import numpy as np
import structlog
from numpy.typing import NDArray
from tradingview_screener import Column, Query

from argus.config import get_settings
from argus.data.prices.base import BAR_DTYPE, ProviderHealth, Quote
from argus.markets import IN_NSE, Instrument, Market
from argus.markets.registry import US_NASDAQ, US_NYSE

logger = structlog.get_logger(__name__)

_TV_MARKET_BY_CODE = {
    US_NYSE.code: "america",
    US_NASDAQ.code: "america",
    IN_NSE.code: "india",
}

# Floor market cap filter for top_liquid -- keeps micro-caps/illiquid shells
# out of the "most liquid" ranking regardless of a temporary volume spike.
_MIN_MARKET_CAP = 100_000_000.0


class UniverseSource(Protocol):
    """Optional capability: a provider that can rank/return the most liquid
    instruments in a market. Implemented here by ``TVScreenerProvider``;
    consumed by ``argus.data.universe.TVUniverseProvider``."""

    async def top_liquid(self, market: Market, n: int) -> list[Instrument]:
        """Return up to ``n`` of the most liquid instruments in ``market``,
        ranked by dollar volume traded. Never raises; empty list on failure."""
        ...


class TVScreenerProvider:
    """Quotes + universe ranking via the TradingView scanner. No OHLCV history."""

    name = "tvscreener"

    def __init__(self, *, timeout_seconds: float | None = None) -> None:
        data_settings = get_settings().data
        self._timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else data_settings.provider_timeout_seconds
        )

    def supports(self, market: Market) -> bool:
        return market.code in _TV_MARKET_BY_CODE

    async def get_daily_bars(
        self, inst: Instrument, start: date, end: date
    ) -> NDArray[np.void]:
        return np.zeros(0, dtype=BAR_DTYPE)

    async def get_quote(self, inst: Instrument) -> Quote | None:
        tv_market = _TV_MARKET_BY_CODE.get(inst.market_code)
        if tv_market is None:
            return None
        try:
            rows = await asyncio.wait_for(
                asyncio.to_thread(self._quote_query_sync, tv_market, [inst.symbol]),
                timeout=self._timeout_seconds,
            )
        except Exception as exc:  # noqa: BLE001 -- provider methods never raise
            logger.warning(
                "tvscreener.get_quote.error",
                symbol=inst.symbol,
                market=inst.market_code,
                error=str(exc) or "timed out",
            )
            return None

        if not rows:
            return None
        return _quote_from_row(inst, rows[0])

    def _quote_query_sync(self, tv_market: str, symbols: list[str]) -> list[dict[str, Any]]:
        query = (
            Query()
            .select("name", "close", "change")
            .where(Column("name").isin(symbols))
            .set_markets(tv_market)
            .limit(len(symbols))
        )
        _count, df = query.get_scanner_data()
        rows: list[dict[str, Any]] = cast("list[dict[str, Any]]", df.to_dict("records"))
        return rows

    async def top_liquid(self, market: Market, n: int) -> list[Instrument]:
        tv_market = _TV_MARKET_BY_CODE.get(market.code)
        if tv_market is None:
            return []
        try:
            rows = await asyncio.wait_for(
                asyncio.to_thread(self._top_liquid_query_sync, tv_market, n),
                timeout=self._timeout_seconds,
            )
        except Exception as exc:  # noqa: BLE001 -- provider methods never raise
            logger.warning(
                "tvscreener.top_liquid.error",
                market=market.code,
                n=n,
                error=str(exc) or "timed out",
            )
            return []

        instruments: list[Instrument] = []
        for row in rows:
            symbol = _to_str(row.get("name"))
            if not symbol:
                continue
            instruments.append(
                Instrument(
                    symbol=symbol,
                    market_code=market.code,
                    name=_to_str(row.get("description")),
                    sector=_to_str(row.get("sector")),
                )
            )
        return instruments

    def _top_liquid_query_sync(self, tv_market: str, n: int) -> list[dict[str, Any]]:
        query = (
            Query()
            .select("name", "description", "sector")
            .where(Column("market_cap_basic") > _MIN_MARKET_CAP)
            .set_markets(tv_market)
            .order_by("Value.Traded", ascending=False)
            .limit(n)
        )
        _count, df = query.get_scanner_data()
        rows: list[dict[str, Any]] = cast("list[dict[str, Any]]", df.to_dict("records"))
        return rows

    async def health_check(self) -> ProviderHealth:
        try:
            rows = await asyncio.wait_for(
                asyncio.to_thread(self._top_liquid_query_sync, "america", 5),
                timeout=self._timeout_seconds,
            )
        except Exception as exc:  # noqa: BLE001 -- provider methods never raise
            return ProviderHealth(
                ok=False, detail=str(exc) or "timed out", checked_at=datetime.now(UTC)
            )
        ok = len(rows) > 0
        detail = f"fetched {len(rows)} rows" if ok else "no rows returned"
        return ProviderHealth(ok=ok, detail=detail, checked_at=datetime.now(UTC))


def _quote_from_row(inst: Instrument, row: dict[str, Any]) -> Quote | None:
    close = _to_float(row.get("close"))
    if close is None:
        return None
    change_pct = _to_float(row.get("change"))
    prev_close = (
        close / (1.0 + change_pct / 100.0)
        if change_pct is not None and change_pct != -100.0
        else None
    )
    return Quote(
        symbol=inst.symbol,
        market_code=inst.market_code,
        price=close,
        prev_close=prev_close,
        ts=datetime.now(UTC),
    )


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN guard -- pandas hands back NaN (not None) for missing numeric cells
        return None
    return f


def _to_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and value != value:  # NaN guard, see _to_float
        return None
    s = str(value)
    return s or None
