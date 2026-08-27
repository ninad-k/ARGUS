"""Momentum ranks strong trend above mild trend; breakout finds a hand-built coil
and skips a mid-range symbol. Entries/stops/targets must be finite and ordered.
"""

from datetime import date, timedelta

import numpy as np
from numpy.typing import NDArray

from argus.data.prices.base import bars_from_columns
from argus.data.prices.static_provider import synthetic_bars
from argus.indicators.features import compute_features
from argus.markets import US_NASDAQ, Instrument, Market
from argus.screener.base import Candidate
from argus.screener.strategies.breakout import BreakoutStrategy
from argus.screener.strategies.momentum import MomentumStrategy


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


def _assert_bracket_ordered_and_finite(c: Candidate) -> None:
    assert c.entry is not None and c.stop is not None and c.target is not None
    assert np.isfinite(c.entry)
    assert np.isfinite(c.stop)
    assert np.isfinite(c.target)
    assert c.stop < c.entry < c.target


def _build_breakout_bars(
    *,
    base: float = 100.0,
    n1: int = 220,
    n2: int = 49,
    tight_range: float = 0.002,
    wide_range: float = 0.015,
    vol: float = 1_000_000.0,
    spike_vol_mult: float = 5.0,
    seed: int = 1,
) -> NDArray[np.void]:
    """A hand-built near-high consolidation: a wide-range uptrend, then a tight
    (low bb-width) consolidation, then a high-volume breakout day."""
    rng = np.random.RandomState(seed)
    trend_pct = np.linspace(0, 0.40, n1)
    closes1 = base * (1 + trend_pct)
    level2 = closes1[-1]
    closes2 = level2 + rng.uniform(-level2 * tight_range / 2, level2 * tight_range / 2, size=n2)
    closes = np.concatenate([closes1, closes2])
    n = len(closes)

    opens = np.empty(n)
    opens[0] = closes[0]
    opens[1:] = closes[:-1]

    highs = np.empty(n)
    lows = np.empty(n)
    highs[:n1] = np.maximum(opens[:n1], closes[:n1]) * (1 + wide_range)
    lows[:n1] = np.minimum(opens[:n1], closes[:n1]) * (1 - wide_range)
    highs[n1:] = np.maximum(opens[n1:], closes[n1:]) * (1 + tight_range)
    lows[n1:] = np.minimum(opens[n1:], closes[n1:]) * (1 - tight_range)
    volumes = np.full(n, vol)

    # Breakout day: closes above the prior range high on a volume spike.
    prior_high = highs.max()
    bo_close = prior_high * 1.02
    bo_open = closes[-1]
    bo_high = bo_close * 1.005
    bo_low = min(bo_open, bo_close) * 0.995
    closes = np.append(closes, bo_close)
    opens = np.append(opens, bo_open)
    highs = np.append(highs, bo_high)
    lows = np.append(lows, bo_low)
    volumes = np.append(volumes, vol * spike_vol_mult)

    n = len(closes)
    start = date(2025, 1, 1)
    ts = np.array(
        [np.datetime64((start + timedelta(days=i)).isoformat(), "s") for i in range(n)],
        dtype="datetime64[s]",
    )
    return bars_from_columns(ts, opens, highs, lows, closes, volumes)


# ---------------------------------------------------------------------------
# MomentumStrategy
# ---------------------------------------------------------------------------


async def test_momentum_ranks_strong_trend_above_mild_trend() -> None:
    strong = synthetic_bars(n=300, start_price=100.0, seed=101, start=date(2025, 1, 2), trend=0.004)
    mild = synthetic_bars(n=300, start_price=100.0, seed=102, start=date(2025, 1, 2), trend=0.0006)
    ctx = _FakeContext(US_NASDAQ, {"STRONG": strong, "MILD": mild})

    candidates = await MomentumStrategy().screen(ctx)
    by_symbol = {c.instrument.symbol: c for c in candidates}

    assert set(by_symbol) == {"STRONG", "MILD"}
    assert by_symbol["STRONG"].score > by_symbol["MILD"].score
    for c in candidates:
        assert c.stage == "uptrend"
        _assert_bracket_ordered_and_finite(c)


async def test_momentum_excludes_symbols_failing_the_trend_gate() -> None:
    strong = synthetic_bars(n=300, start_price=100.0, seed=101, start=date(2025, 1, 2), trend=0.004)
    flat = synthetic_bars(n=300, start_price=100.0, seed=103, start=date(2025, 1, 2), trend=0.0)
    ctx = _FakeContext(US_NASDAQ, {"STRONG": strong, "FLAT": flat})

    candidates = await MomentumStrategy().screen(ctx)
    symbols = {c.instrument.symbol for c in candidates}

    assert "STRONG" in symbols
    assert "FLAT" not in symbols  # doesn't clear close > sma_50 > sma_200


async def test_momentum_empty_universe_returns_no_candidates() -> None:
    ctx = _FakeContext(US_NASDAQ, {})
    assert await MomentumStrategy().screen(ctx) == []


# ---------------------------------------------------------------------------
# BreakoutStrategy
# ---------------------------------------------------------------------------


async def test_breakout_detects_near_high_contraction_and_skips_mid_range() -> None:
    coiled = _build_breakout_bars()
    mid_range = synthetic_bars(
        n=270, start_price=100.0, seed=55, start=date(2025, 1, 1), trend=0.0005
    )
    ctx = _FakeContext(US_NASDAQ, {"COILED": coiled, "MIDRANGE": mid_range})

    candidates = await BreakoutStrategy().screen(ctx)
    symbols = {c.instrument.symbol: c for c in candidates}

    assert "COILED" in symbols
    assert "MIDRANGE" not in symbols  # nowhere near its 252d high

    coiled_candidate = symbols["COILED"]
    assert coiled_candidate.stage in {"breakout", "pre-breakout"}
    assert "contract" in coiled_candidate.reason
    _assert_bracket_ordered_and_finite(coiled_candidate)


async def test_breakout_marks_stage_breakout_when_closing_above_prior_high() -> None:
    coiled = _build_breakout_bars()
    ctx = _FakeContext(US_NASDAQ, {"COILED": coiled})

    candidates = await BreakoutStrategy().screen(ctx)
    assert len(candidates) == 1
    # The hand-built fixture's final bar closes above every prior high.
    assert candidates[0].stage == "breakout"


async def test_breakout_empty_universe_returns_no_candidates() -> None:
    ctx = _FakeContext(US_NASDAQ, {})
    assert await BreakoutStrategy().screen(ctx) == []
