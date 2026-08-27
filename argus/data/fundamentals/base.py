"""Fundamentals-data provider protocol and shared types.

Mirrors the DRUVA fundamentals-provider idiom (Protocol + Null/Static/HTTP-ish
implementations selected via a factory -- see
``DRUVA/backend/app/core/advisor/valuation/fundamentals_provider.py``) but
scoped to the ratio-only view ARGUS's screener strategies need: no historical
line items, no DCF support -- just the latest snapshot of valuation/quality
ratios a value-style strategy filters or scores on.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Protocol

from argus.markets import Instrument, Market

if TYPE_CHECKING:
    from collections.abc import Sequence

# Bounds how many concurrent per-symbol lookups ``default_get_many`` issues --
# relevant for providers with no bulk endpoint (e.g. yfinance's per-symbol
# ``Ticker.info``) so a large universe doesn't fire hundreds of requests at once.
_DEFAULT_GET_MANY_CONCURRENCY = 5


@dataclass(frozen=True, slots=True)
class FundamentalsView:
    """A single point-in-time fundamentals snapshot for one instrument.

    Ratio fields are fractions, not percentages, unless noted otherwise (e.g.
    ``roe=0.15`` means 15% ROE) -- callers should not need to know which
    upstream provider produced a given view.
    """

    symbol: str
    market_code: str
    as_of: date
    market_cap: float | None = None
    pe: float | None = None
    forward_pe: float | None = None
    pb: float | None = None
    ps: float | None = None
    roe: float | None = None
    debt_to_equity: float | None = None
    revenue_growth: float | None = None
    earnings_growth: float | None = None
    dividend_yield: float | None = None
    profit_margin: float | None = None
    sector: str | None = None


class FundamentalsProvider(Protocol):
    """Async fundamentals lookup. Implementations must never raise -- they
    log and return ``None``/an empty dict on failure."""

    name: str

    def supports(self, market: Market) -> bool:
        """Whether this provider can serve fundamentals for ``market``."""
        ...

    async def get(self, inst: Instrument) -> FundamentalsView | None:
        """Return the latest fundamentals view for ``inst``, or ``None``."""
        ...

    async def get_many(self, insts: list[Instrument]) -> dict[str, FundamentalsView]:
        """Return fundamentals for as many of ``insts`` as are available,
        keyed by symbol. Missing/failed lookups are simply absent from the
        result -- never raises."""
        ...


async def default_get_many(
    provider: FundamentalsProvider, insts: Sequence[Instrument]
) -> dict[str, FundamentalsView]:
    """Default ``get_many``: calls ``provider.get()`` for each instrument
    with bounded concurrency.

    Suitable for providers with no bulk endpoint. Providers with a genuine
    bulk API (e.g. ``TVScreenerFundamentalsProvider``) should override
    ``get_many`` with a real batched query instead of using this.
    """
    semaphore = asyncio.Semaphore(_DEFAULT_GET_MANY_CONCURRENCY)
    results: dict[str, FundamentalsView] = {}

    async def _one(inst: Instrument) -> None:
        async with semaphore:
            view = await provider.get(inst)
        if view is not None:
            results[inst.symbol] = view

    await asyncio.gather(*(_one(inst) for inst in insts))
    return results


class NullFundamentalsProvider:
    """Always returns ``None``/empty. Used when no fundamentals source is configured."""

    name = "null"

    def supports(self, market: Market) -> bool:
        return True

    async def get(self, inst: Instrument) -> FundamentalsView | None:
        return None

    async def get_many(self, insts: list[Instrument]) -> dict[str, FundamentalsView]:
        return {}


class StaticFundamentalsProvider:
    """Serves fundamentals from a fixed in-memory map, keyed by symbol.
    Primarily for tests."""

    name = "static"

    def __init__(self, views: dict[str, FundamentalsView] | None = None) -> None:
        self._views: dict[str, FundamentalsView] = views or {}

    def add(self, view: FundamentalsView) -> None:
        self._views[view.symbol] = view

    def supports(self, market: Market) -> bool:
        return True

    async def get(self, inst: Instrument) -> FundamentalsView | None:
        return self._views.get(inst.symbol)

    async def get_many(self, insts: list[Instrument]) -> dict[str, FundamentalsView]:
        return await default_get_many(self, insts)
