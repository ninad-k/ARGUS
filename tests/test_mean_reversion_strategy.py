"""MeanReversionStrategy: gates (non-uptrend, non-oversold, roc_60 turned
negative, volume collapsed all rejected), deeper-oversold-scores-higher
ordering, and bracket sanity.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
from numpy.typing import NDArray

from argus.data.prices.base import bars_from_columns
from argus.data.prices.static_provider import synthetic_bars
from argus.indicators.features import compute_features
from argus.markets import US_NASDAQ, Instrument, Market
from argus.screener.base import Candidate
from argus.screener.strategies.mean_reversion import MeanReversionStrategy


class _FakeContext:
    """Minimal ``ScreenContext`` over an in-memory bars map -- no BarStore/DB needed."""

    def __init__(self, market: Market, bars_by_symbol: dict[str, NDArray[np.void]]) -> None:
        self.market = market
        self._bars = bars_by_symbol
        self._instruments = [
            Instrument(symbol=sym, market_code=market.code) for sym in bars_by_symbol
        ]
        self._feature_cache: dict[str, dict[str, float]] = {}

    async def universe(self) -> list[Instrument]:
        return list(self._instruments)

    async def bars(self, inst: Instrument, n: int = 260) -> NDArray[np.void]:
        arr = self._bars[inst.symbol]
        return arr[-n:] if n < len(arr) else arr

    async def features(self, inst: Instrument) -> dict[str, float]:
        if inst.symbol not in self._feature_cache:
            self._feature_cache[inst.symbol] = compute_features(self._bars[inst.symbol])
        return self._feature_cache[inst.symbol]


def _build_pullback_bars(
    *,
    n1: int = 250,
    n2: int = 10,
    uptrend_daily: float = 0.008,
    pullback_daily: float = -0.02,
    vol: float = 1_000_000.0,
    seed: int = 1,
    base: float = 100.0,
) -> NDArray[np.void]:
    """A steady uptrend of ``n1`` days followed by a sharper ``n2``-day
    pullback -- tunable to land in/out of the strategy's gate."""
    rng = np.random.RandomState(seed)
    r1 = rng.normal(loc=uptrend_daily, scale=0.005, size=n1)
    c1 = base * np.cumprod(1.0 + r1)
    r2 = rng.normal(loc=pullback_daily, scale=0.005, size=n2)
    c2 = c1[-1] * np.cumprod(1.0 + r2)
    closes = np.concatenate([c1, c2])
    n = len(closes)

    opens = np.empty(n)
    opens[0] = closes[0]
    opens[1:] = closes[:-1]
    intraday = np.abs(rng.normal(loc=0.0, scale=0.004, size=n))
    highs = np.maximum(opens, closes) * (1 + intraday)
    lows = np.minimum(opens, closes) * (1 - intraday)
    volumes = np.full(n, vol)

    start = date(2025, 1, 1)
    ts = np.array(
        [np.datetime64((start + timedelta(days=i)).isoformat(), "s") for i in range(n)],
        dtype="datetime64[s]",
    )
    return bars_from_columns(ts, opens, highs, lows, closes, volumes)


# Verified via `compute_features`: rsi_14 ~26 (oversold), roc_60 ~+20%
# (uptrend intact), close < sma_20 (pulled back), close > sma_200, rvol ~1.0.
def _passing_pullback_bars(seed: int = 1) -> NDArray[np.void]:
    return _build_pullback_bars(n2=10, pullback_daily=-0.02, uptrend_daily=0.008, seed=seed)


def _assert_bracket_ordered_and_finite(c: Candidate) -> None:
    assert c.entry is not None and c.stop is not None and c.target is not None
    assert np.isfinite(c.entry)
    assert np.isfinite(c.stop)
    assert np.isfinite(c.target)
    assert c.stop < c.entry < c.target


async def test_mean_reversion_excludes_non_uptrend() -> None:
    downtrend = synthetic_bars(
        n=300, start_price=100.0, seed=301, start=date(2025, 1, 1), trend=-0.01
    )
    ctx = _FakeContext(US_NASDAQ, {"DOWNTREND": downtrend})
    candidates = await MeanReversionStrategy().screen(ctx)
    assert candidates == []


async def test_mean_reversion_excludes_non_oversold() -> None:
    steady_uptrend = synthetic_bars(
        n=300, start_price=100.0, seed=302, start=date(2025, 1, 1), trend=0.002
    )
    ctx = _FakeContext(US_NASDAQ, {"STEADY": steady_uptrend})
    candidates = await MeanReversionStrategy().screen(ctx)
    assert candidates == []


async def test_mean_reversion_excludes_broken_longer_term_momentum() -> None:
    """Oversold and pulled back, but the 60-day trend has actually turned
    negative -- roc_60 <= 0 means this isn't "pullback in an uptrend"
    anymore, it's just a downtrend."""
    long_pullback = _build_pullback_bars(n2=50, pullback_daily=-0.008, uptrend_daily=0.008, seed=1)
    features = compute_features(long_pullback)
    assert features["rsi_14"] < 35  # oversold...
    assert features["roc_60"] < 0  # ...but longer-term momentum has broken

    ctx = _FakeContext(US_NASDAQ, {"BROKEN": long_pullback})
    candidates = await MeanReversionStrategy().screen(ctx)
    assert candidates == []


async def test_mean_reversion_excludes_collapsed_volume() -> None:
    bars = _passing_pullback_bars().copy()
    bars["volume"][-1] = bars["volume"][-1] * 0.01  # tape has dried up
    features = compute_features(bars)
    assert features["rvol"] < 0.5

    ctx = _FakeContext(US_NASDAQ, {"DRIEDUP": bars})
    candidates = await MeanReversionStrategy().screen(ctx)
    assert candidates == []


async def test_mean_reversion_passes_gate_and_sets_stage() -> None:
    bars = _passing_pullback_bars()
    ctx = _FakeContext(US_NASDAQ, {"OVERSOLD": bars})
    candidates = await MeanReversionStrategy().screen(ctx)
    assert len(candidates) == 1
    assert candidates[0].stage == "pullback"
    _assert_bracket_ordered_and_finite(candidates[0])


async def test_mean_reversion_deeper_oversold_scores_higher() -> None:
    mild = _build_pullback_bars(n2=10, pullback_daily=-0.015, uptrend_daily=0.008, seed=1)
    deep = _build_pullback_bars(n2=9, pullback_daily=-0.03, uptrend_daily=0.008, seed=1)
    mild_features = compute_features(mild)
    deep_features = compute_features(deep)
    assert deep_features["rsi_14"] < mild_features["rsi_14"]  # deep is genuinely more oversold

    ctx = _FakeContext(US_NASDAQ, {"MILD": mild, "DEEP": deep})
    candidates = await MeanReversionStrategy().screen(ctx)
    by_symbol = {c.instrument.symbol: c for c in candidates}

    assert set(by_symbol) == {"MILD", "DEEP"}
    assert by_symbol["DEEP"].score > by_symbol["MILD"].score
    for c in candidates:
        _assert_bracket_ordered_and_finite(c)


async def test_mean_reversion_empty_universe_returns_no_candidates() -> None:
    ctx = _FakeContext(US_NASDAQ, {})
    assert await MeanReversionStrategy().screen(ctx) == []
