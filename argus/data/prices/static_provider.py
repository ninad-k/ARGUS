"""In-memory price provider for tests and offline development."""

from datetime import UTC, date, datetime, timedelta

import numpy as np
from numpy.typing import NDArray

from argus.data.prices.base import BAR_DTYPE, ProviderHealth, Quote, bars_from_columns
from argus.markets import Instrument, Market


def synthetic_bars(
    n: int,
    start_price: float,
    seed: int,
    start: date,
    trend: float = 0.0,
) -> NDArray[np.void]:
    """Deterministic synthetic daily OHLCV bars for ``n`` trading days.

    Same ``seed`` always produces byte-identical output — used throughout the
    test suite as a stand-in for real market data.
    """
    rng = np.random.RandomState(seed)
    daily_returns = rng.normal(loc=trend, scale=0.01, size=n)
    closes = start_price * np.cumprod(1.0 + daily_returns)

    opens = np.empty(n, dtype=np.float64)
    opens[0] = start_price
    opens[1:] = closes[:-1]

    intraday_range = np.abs(rng.normal(loc=0.0, scale=0.005, size=n))
    highs = np.maximum(opens, closes) * (1.0 + intraday_range)
    lows = np.minimum(opens, closes) * (1.0 - intraday_range)
    volumes = rng.randint(100_000, 5_000_000, size=n).astype(np.float64)

    dates = [start + timedelta(days=i) for i in range(n)]
    ts = np.array(
        [np.datetime64(datetime(d.year, d.month, d.day), "s") for d in dates],
        dtype="datetime64[s]",
    )

    return bars_from_columns(ts, opens, highs, lows, closes, volumes)


class StaticPriceProvider:
    """Serves bars from a fixed in-memory map. Supports all markets."""

    name = "static"

    def __init__(self, bars_by_symbol: dict[str, NDArray[np.void]] | None = None) -> None:
        self._bars: dict[str, NDArray[np.void]] = bars_by_symbol or {}

    def add(self, symbol: str, bars: NDArray[np.void]) -> None:
        self._bars[symbol] = bars

    def supports(self, market: Market) -> bool:
        return True

    async def get_daily_bars(
        self, inst: Instrument, start: date, end: date
    ) -> NDArray[np.void]:
        bars = self._bars.get(inst.symbol)
        if bars is None or len(bars) == 0:
            return np.zeros(0, dtype=BAR_DTYPE)
        start_ts = np.datetime64(start, "s")
        end_ts = np.datetime64(end, "s") + np.timedelta64(1, "D")
        mask = (bars["ts"] >= start_ts) & (bars["ts"] < end_ts)
        return bars[mask]

    async def get_quote(self, inst: Instrument) -> Quote | None:
        bars = self._bars.get(inst.symbol)
        if bars is None or len(bars) == 0:
            return None
        last = bars[-1]
        prev_close = float(bars[-2]["close"]) if len(bars) >= 2 else None
        ts = last["ts"].astype("datetime64[s]").astype(datetime).replace(tzinfo=UTC)
        return Quote(
            symbol=inst.symbol,
            market_code=inst.market_code,
            price=float(last["close"]),
            prev_close=prev_close,
            ts=ts,
        )

    async def health_check(self) -> ProviderHealth:
        return ProviderHealth(
            ok=True, detail="static provider always healthy", checked_at=datetime.now(UTC)
        )
