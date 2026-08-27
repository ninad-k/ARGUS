"""OrderflowConfluenceStrategy: a fixture engineering >=2 confluences produces
a candidate with a sane (ordered, finite) bracket; a fixture with only 1
confluence produces none; the registry discovers all 5 built-in strategies."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
from numpy.typing import NDArray

from argus.data.prices.base import bars_from_columns
from argus.indicators.features import compute_features
from argus.markets import US_NASDAQ, Instrument, Market
from argus.orderflow.features import OrderflowFeatures, compute_orderflow
from argus.screener.registry import all_strategies
from argus.screener.strategies.orderflow_confluence import OrderflowConfluenceStrategy

_START = date(2026, 1, 2)


class _FakeContext:
    """Minimal ``ScreenContext`` over an in-memory bars map, mirroring
    ``tests/test_strategies.py``'s ``_FakeContext`` but with an
    ``orderflow()`` method too (real ``compute_orderflow`` over the same
    fixture bars, not a stub -- this exercises the actual OHLCV-derived
    confluence math, not a canned answer)."""

    def __init__(self, market: Market, bars_by_symbol: dict[str, NDArray[np.void]]) -> None:
        self.market = market
        self._bars = bars_by_symbol
        self._instruments = [
            Instrument(symbol=sym, market_code=market.code) for sym in bars_by_symbol
        ]

    async def universe(self) -> list[Instrument]:
        return list(self._instruments)

    async def bars(self, inst: Instrument, n: int = 260) -> NDArray[np.void]:
        arr = self._bars[inst.symbol]
        return arr[-n:] if n < len(arr) else arr

    async def features(self, inst: Instrument) -> dict[str, float]:
        return compute_features(self._bars[inst.symbol])

    async def orderflow(self, inst: Instrument) -> OrderflowFeatures | None:
        return compute_orderflow(self._bars[inst.symbol])


def _build_bars(
    *, seed: int, n: int = 30, base: float = 100.0, rig_confluences: bool
) -> NDArray[np.void]:
    """A mild-uptrend fixture. With ``rig_confluences=True``, the last bar is
    reshaped into an unfilled gap-up on elevated volume -- on top of the
    close-above-POC confluence every draw of this random walk tends to
    produce naturally, that's 2 (often 3, with the prior-week-high reclaim)
    confluences. With ``rig_confluences=False`` the last bar is left as an
    ordinary trend bar, leaving only the close-above-POC confluence."""
    rng = np.random.RandomState(seed)
    daily_returns = rng.normal(loc=0.003, scale=0.01, size=n)
    closes = base * np.cumprod(1.0 + daily_returns)
    opens = np.empty(n)
    opens[0] = base
    opens[1:] = closes[:-1]
    intraday_range = np.abs(rng.normal(loc=0.0, scale=0.005, size=n))
    highs = np.maximum(opens, closes) * (1.0 + intraday_range)
    lows = np.minimum(opens, closes) * (1.0 - intraday_range)
    volumes = rng.randint(100_000, 500_000, size=n).astype(np.float64)

    if rig_confluences:
        prev_close = closes[-2]
        opens[-1] = prev_close * 1.02
        closes[-1] = opens[-1] * 1.01
        lows[-1] = opens[-1] * 0.999  # stays above prev_close -- unfilled gap
        highs[-1] = closes[-1] * 1.005
        volumes[-1] = float(np.mean(volumes[:-1])) * 3  # elevated rvol

    ts = np.array(
        [np.datetime64((_START + timedelta(days=i)).isoformat(), "s") for i in range(n)],
        dtype="datetime64[s]",
    )
    return bars_from_columns(ts, opens, highs, lows, closes, volumes)


async def test_two_plus_confluences_produce_a_candidate_with_sane_bracket() -> None:
    bars = _build_bars(seed=3, rig_confluences=True)
    ctx = _FakeContext(US_NASDAQ, {"CONFLUENT": bars})

    candidates = await OrderflowConfluenceStrategy().screen(ctx)

    assert len(candidates) == 1
    c = candidates[0]
    assert c.instrument.symbol == "CONFLUENT"
    assert c.stage == "orderflow"
    assert c.direction == "long"
    assert c.entry is not None and c.stop is not None and c.target is not None
    assert np.isfinite(c.entry)
    assert np.isfinite(c.stop)
    assert np.isfinite(c.target)
    assert c.stop < c.entry < c.target
    assert c.reason.startswith(("2 orderflow confluences", "3 orderflow confluences"))


async def test_single_confluence_produces_no_candidate() -> None:
    bars = _build_bars(seed=3, rig_confluences=False)
    ctx = _FakeContext(US_NASDAQ, {"WEAK": bars})

    candidates = await OrderflowConfluenceStrategy().screen(ctx)

    assert candidates == []


async def test_empty_universe_returns_no_candidates() -> None:
    ctx = _FakeContext(US_NASDAQ, {})
    assert await OrderflowConfluenceStrategy().screen(ctx) == []


def test_registry_discovers_five_strategies() -> None:
    assert set(all_strategies()) == {
        "momentum",
        "breakout",
        "value",
        "mean_reversion",
        "orderflow_confluence",
    }
