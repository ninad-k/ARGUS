"""Price-data provider protocol and shared types.

Bars are carried as numpy structured arrays (``BAR_DTYPE``) rather than a
list of dataclasses — this keeps the hot path (feature computation over
hundreds of symbols x years of daily bars) vectorized and cheap to store in
DuckDB/Parquet.
"""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from argus.markets import Instrument, Market

BAR_DTYPE = np.dtype(
    [
        ("ts", "datetime64[s]"),
        ("open", "f8"),
        ("high", "f8"),
        ("low", "f8"),
        ("close", "f8"),
        ("volume", "f8"),
    ]
)


def bars_from_columns(
    ts: NDArray[np.datetime64],
    open_: NDArray[np.float64],
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    volume: NDArray[np.float64],
) -> NDArray[np.void]:
    """Build a ``BAR_DTYPE`` structured array from parallel column arrays."""
    n = len(ts)
    bars = np.zeros(n, dtype=BAR_DTYPE)
    bars["ts"] = ts
    bars["open"] = open_
    bars["high"] = high
    bars["low"] = low
    bars["close"] = close
    bars["volume"] = volume
    return bars


@dataclass(frozen=True, slots=True)
class Quote:
    """A single point-in-time price quote."""

    symbol: str
    market_code: str
    price: float
    prev_close: float | None
    ts: datetime


@dataclass(frozen=True, slots=True)
class ProviderHealth:
    """Result of a provider self-check."""

    ok: bool
    detail: str
    checked_at: datetime


class PriceDataProvider(Protocol):
    """Async OHLCV price-data source. Implementations must never raise —
    they log and return an empty result on failure."""

    name: str

    def supports(self, market: Market) -> bool:
        """Whether this provider can serve data for ``market``."""
        ...

    async def get_daily_bars(
        self, inst: Instrument, start: date, end: date
    ) -> NDArray[np.void]:
        """Return daily OHLCV bars for ``[start, end]`` inclusive, ``BAR_DTYPE``,
        ascending by ``ts``. May be empty; never raises."""
        ...

    async def get_quote(self, inst: Instrument) -> Quote | None:
        """Return the latest available quote, or ``None`` if unavailable."""
        ...

    async def health_check(self) -> ProviderHealth:
        """Cheap self-check used by the source registry."""
        ...
