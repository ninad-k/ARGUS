"""ValueStrategy: fundamentals gate (market cap by market, PE band, ROE floor,
debt/equity ceiling, missing-fundamentals rejection, downtrend value-trap
guard), cheap+quality-beats-expensive+junk scoring order, and bracket sanity.
"""

from __future__ import annotations

from datetime import date

import numpy as np
from numpy.typing import NDArray

from argus.data.fundamentals import FundamentalsView
from argus.data.prices.static_provider import synthetic_bars
from argus.indicators.features import compute_features
from argus.markets import IN_NSE, US_NASDAQ, Instrument, Market
from argus.screener.base import Candidate
from argus.screener.registry import all_strategies
from argus.screener.strategies.value import ValueStrategy


class _FakeContext:
    """Minimal ``ScreenContext`` over in-memory bars + fundamentals maps."""

    def __init__(
        self,
        market: Market,
        bars_by_symbol: dict[str, NDArray[np.void]],
        fundamentals_by_symbol: dict[str, FundamentalsView] | None = None,
    ) -> None:
        self.market = market
        self._bars = bars_by_symbol
        self._fundamentals = fundamentals_by_symbol or {}
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

    async def fundamentals(self, inst: Instrument) -> FundamentalsView | None:
        return self._fundamentals.get(inst.symbol)


def _fv(
    symbol: str,
    market_code: str = US_NASDAQ.code,
    *,
    market_cap: float | None = 5_000_000_000.0,
    pe: float | None = 15.0,
    pb: float | None = 3.0,
    roe: float | None = 0.20,
    debt_to_equity: float | None = 0.5,
    revenue_growth: float | None = 0.08,
    earnings_growth: float | None = 0.10,
    profit_margin: float | None = 0.15,
) -> FundamentalsView:
    return FundamentalsView(
        symbol=symbol,
        market_code=market_code,
        as_of=date(2025, 1, 1),
        market_cap=market_cap,
        pe=pe,
        pb=pb,
        roe=roe,
        debt_to_equity=debt_to_equity,
        revenue_growth=revenue_growth,
        earnings_growth=earnings_growth,
        profit_margin=profit_margin,
    )


# A mild, noisy uptrend whose close ends up only a shallow break below its own
# sma200 (dist_sma200_pct ~ -5.7%, within the -10% value-trap guard) -- used
# as the standard "passes the technical gate" fixture across tests below.
def _ok_bars(seed: int = 201) -> NDArray[np.void]:
    return synthetic_bars(n=300, start_price=100.0, seed=seed, start=date(2025, 1, 1), trend=0.0015)


def _deep_downtrend_bars(seed: int = 202) -> NDArray[np.void]:
    return synthetic_bars(n=300, start_price=100.0, seed=seed, start=date(2025, 1, 1), trend=-0.01)


def _assert_bracket_ordered_and_finite(c: Candidate) -> None:
    assert c.entry is not None and c.stop is not None and c.target is not None
    assert np.isfinite(c.entry)
    assert np.isfinite(c.stop)
    assert np.isfinite(c.target)
    assert c.stop < c.entry < c.target


async def test_value_gate_rejects_high_pe() -> None:
    bars = _ok_bars()
    ctx = _FakeContext(
        US_NASDAQ, {"HIGHPE": bars}, {"HIGHPE": _fv("HIGHPE", pe=55.0)}
    )
    candidates = await ValueStrategy().screen(ctx)
    assert candidates == []


async def test_value_gate_rejects_low_roe() -> None:
    bars = _ok_bars()
    ctx = _FakeContext(US_NASDAQ, {"LOWROE": bars}, {"LOWROE": _fv("LOWROE", roe=0.05)})
    candidates = await ValueStrategy().screen(ctx)
    assert candidates == []


async def test_value_gate_rejects_high_debt_to_equity() -> None:
    bars = _ok_bars()
    ctx = _FakeContext(
        US_NASDAQ, {"LEVERED": bars}, {"LEVERED": _fv("LEVERED", debt_to_equity=3.0)}
    )
    candidates = await ValueStrategy().screen(ctx)
    assert candidates == []


async def test_value_gate_allows_missing_debt_to_equity() -> None:
    """``debt_to_equity`` unknown -> that sub-gate is skipped, not failed."""
    bars = _ok_bars()
    ctx = _FakeContext(
        US_NASDAQ, {"NODATA": bars}, {"NODATA": _fv("NODATA", debt_to_equity=None)}
    )
    candidates = await ValueStrategy().screen(ctx)
    assert {c.instrument.symbol for c in candidates} == {"NODATA"}


async def test_value_gate_rejects_no_fundamentals() -> None:
    bars = _ok_bars()
    ctx = _FakeContext(US_NASDAQ, {"NOFUND": bars}, {})
    candidates = await ValueStrategy().screen(ctx)
    assert candidates == []


async def test_value_gate_rejects_downtrend_value_trap() -> None:
    """Deep below sma200 (dist_sma200_pct well past -10%) -> rejected even
    though the fundamentals are otherwise pristine -- cheap-and-still-falling
    is a value trap, not a value pick."""
    bars = _deep_downtrend_bars()
    ctx = _FakeContext(US_NASDAQ, {"TRAP": bars}, {"TRAP": _fv("TRAP")})
    candidates = await ValueStrategy().screen(ctx)
    assert candidates == []


async def test_value_gate_market_cap_threshold_differs_by_market() -> None:
    bars = _ok_bars()

    us_small = _FakeContext(
        US_NASDAQ, {"SMALL": bars}, {"SMALL": _fv("SMALL", market_cap=500_000_000.0)}
    )
    assert await ValueStrategy().screen(us_small) == []

    us_large = _FakeContext(
        US_NASDAQ, {"LARGE": bars}, {"LARGE": _fv("LARGE", market_cap=3_000_000_000.0)}
    )
    assert {c.instrument.symbol for c in await ValueStrategy().screen(us_large)} == {"LARGE"}

    in_small = _FakeContext(
        IN_NSE,
        {"INSMALL": bars},
        {"INSMALL": _fv("INSMALL", IN_NSE.code, market_cap=50_000_000_000.0)},
    )
    assert await ValueStrategy().screen(in_small) == []

    in_large = _FakeContext(
        IN_NSE,
        {"INLARGE": bars},
        {"INLARGE": _fv("INLARGE", IN_NSE.code, market_cap=150_000_000_000.0)},
    )
    assert {c.instrument.symbol for c in await ValueStrategy().screen(in_large)} == {"INLARGE"}


async def test_value_scoring_cheap_quality_beats_expensive_junk() -> None:
    bars = _ok_bars()
    cheap = _fv(
        "CHEAP",
        pe=10.0,
        pb=1.5,
        roe=0.25,
        debt_to_equity=0.3,
        revenue_growth=0.15,
        earnings_growth=0.20,
        profit_margin=0.20,
    )
    expensive = _fv(
        "EXPENSIVE",
        pe=35.0,
        pb=8.0,
        roe=0.11,
        debt_to_equity=1.8,
        revenue_growth=0.01,
        earnings_growth=-0.02,
        profit_margin=0.05,
    )
    ctx = _FakeContext(
        US_NASDAQ, {"CHEAP": bars, "EXPENSIVE": bars}, {"CHEAP": cheap, "EXPENSIVE": expensive}
    )

    candidates = await ValueStrategy().screen(ctx)
    by_symbol = {c.instrument.symbol: c for c in candidates}

    assert set(by_symbol) == {"CHEAP", "EXPENSIVE"}
    assert by_symbol["CHEAP"].score > by_symbol["EXPENSIVE"].score
    for c in candidates:
        assert c.stage == "value"
        _assert_bracket_ordered_and_finite(c)


async def test_value_empty_universe_returns_no_candidates() -> None:
    ctx = _FakeContext(US_NASDAQ, {}, {})
    assert await ValueStrategy().screen(ctx) == []


def test_registry_discovers_five_strategies() -> None:
    assert set(all_strategies()) == {
        "momentum",
        "breakout",
        "value",
        "mean_reversion",
        "orderflow_confluence",
    }
