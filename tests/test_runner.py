"""End-to-end run_screen over a StaticPriceProvider-seeded BarStore, plus the
persist_screen_result round-trip into a tmp SQLite control-plane DB.
"""

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from sqlalchemy import select

from argus.config import AppSettings
from argus.data.prices.base import bars_from_columns
from argus.data.prices.static_provider import StaticPriceProvider, synthetic_bars
from argus.data.store.duckdb_ohlcv import BarStore, refresh_bars
from argus.data.universe import StaticUniverseProvider
from argus.db import async_session, init_db
from argus.db.models import DailyPick, ScreenRun
from argus.indicators.features import compute_features
from argus.markets import US_NASDAQ, Instrument
from argus.screener.base import DefaultScreenContext
from argus.screener.filters import build_default_chain
from argus.screener.runner import persist_screen_result, run_screen
from argus.screener.strategies.breakout import BreakoutStrategy
from argus.screener.strategies.momentum import MomentumStrategy

_TODAY = date.today()  # noqa: DTZ011 -- matches refresh_bars' own daily-cache boundary


def _coiled_bars(
    end: date,
    *,
    seed: int = 1,
    n1: int = 220,
    n2: int = 49,
    base: float = 100.0,
    tight_range: float = 0.002,
    wide_range: float = 0.015,
    vol: float = 1_000_000.0,
    last_day_vol_mult: float = 1.5,
    total_gain: float = 0.40,
) -> NDArray[np.void]:
    """A wide-range uptrend followed by a tight consolidation whose last bar
    sits at the top of its own range on above-average volume -- shaped to be
    picked up by *both* MomentumStrategy and BreakoutStrategy at once, so the
    runner's fusion path has something to fuse."""
    rng = np.random.RandomState(seed)
    trend_pct = np.linspace(0, total_gain, n1)
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
    volumes[-1] *= last_day_vol_mult

    # Last bar sits at the top of the consolidation range -> within 3% of the
    # 252d high without the RSI-spiking single-day jump a true breakout gap
    # would cause (which would fail momentum's rsi_14 < 80 gate).
    closes[-1] = closes2.max() * 1.001
    highs[-1] = max(highs[-1], closes[-1] * (1 + tight_range))

    start = end - timedelta(days=n - 1)
    ts = np.array(
        [np.datetime64((start + timedelta(days=i)).isoformat(), "s") for i in range(n)],
        dtype="datetime64[s]",
    )
    return bars_from_columns(ts, opens, highs, lows, closes, volumes)


async def _seed_store(store: BarStore) -> list[Instrument]:
    """Six symbols: a fusion candidate (both strategies), a momentum-only
    candidate, one that clears filters but no strategy gate, and three that
    fail the default filter chain for a different reason each."""
    lookback = 299
    start = _TODAY - timedelta(days=lookback)

    bars_by_symbol = {
        "FUSION": _coiled_bars(_TODAY, seed=1),
        "MOMO": synthetic_bars(n=300, start_price=100.0, seed=201, start=start, trend=0.004),
        "WEAK": synthetic_bars(n=300, start_price=100.0, seed=202, start=start, trend=0.0),
        "PENNY": synthetic_bars(n=300, start_price=2.0, seed=203, start=start, trend=0.001),
        "THIN": synthetic_bars(n=300, start_price=100.0, seed=204, start=start, trend=0.001),
        "SHORTHIST": synthetic_bars(
            n=50, start_price=100.0, seed=205, start=_TODAY - timedelta(days=49), trend=0.001
        ),
    }
    bars_by_symbol["THIN"]["volume"] = 50_000.0  # deterministic low-liquidity rejection

    provider = StaticPriceProvider(bars_by_symbol)
    instruments = [
        Instrument(symbol=symbol, market_code=US_NASDAQ.code) for symbol in bars_by_symbol
    ]
    for inst in instruments:
        symbol_lookback = 49 if inst.symbol == "SHORTHIST" else lookback
        await refresh_bars(store, provider, inst, symbol_lookback)
    return instruments


async def test_run_screen_end_to_end_ranks_and_fuses(tmp_path: Path) -> None:
    with BarStore(tmp_path / "bars.duckdb") as store:
        instruments = await _seed_store(store)
        universe_provider = StaticUniverseProvider({US_NASDAQ.code: instruments})

        result = await run_screen(
            US_NASDAQ, store=store, universe_provider=universe_provider, top_n=5
        )

    assert result.market_code == US_NASDAQ.code
    assert result.universe_size == 6
    assert result.filtered_size == 3  # FUSION, MOMO, WEAK clear the filter chain

    assert result.rejections["PENNY"].startswith("price")
    assert "volume" in result.rejections["THIN"]
    assert "history" in result.rejections["SHORTHIST"]

    by_symbol = {c.instrument.symbol: c for c in result.candidates}
    assert set(by_symbol) == {"FUSION", "MOMO"}  # WEAK clears filters but no strategy gate
    assert result.top == result.candidates[:5]

    # Ranked descending by score.
    scores = [c.score for c in result.candidates]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 100.0 for s in scores)

    fusion = by_symbol["FUSION"]
    assert "momentum" in fusion.strategy
    assert "breakout" in fusion.strategy
    assert "+" in fusion.strategy
    assert "[momentum]" in fusion.reason
    assert "[breakout]" in fusion.reason

    momo = by_symbol["MOMO"]
    assert momo.strategy == "momentum"  # picked by momentum only, no fusion bonus


async def test_run_screen_fusion_bonus_beats_either_solo_score(tmp_path: Path) -> None:
    with BarStore(tmp_path / "bars.duckdb") as store:
        instruments = await _seed_store(store)
        universe_provider = StaticUniverseProvider({US_NASDAQ.code: instruments})
        result = await run_screen(
            US_NASDAQ, store=store, universe_provider=universe_provider, top_n=5
        )

        # Re-derive FUSION's un-fused per-strategy scores from the same
        # filtered universe/context run_screen would have built internally,
        # to check the fused score against the documented bonus formula.
        features_by_symbol = {}
        for inst in instruments:
            bars = store.get_bars(inst.market_code, inst.symbol, 260)
            if len(bars) > 0:
                features_by_symbol[inst.symbol] = compute_features(bars)
        chain = build_default_chain(US_NASDAQ)
        passed, _ = chain.run(instruments, features_by_symbol)
        ctx = DefaultScreenContext(US_NASDAQ, passed, store, feature_cache=features_by_symbol)

        momentum_by_symbol = {c.instrument.symbol: c for c in await MomentumStrategy().screen(ctx)}
        breakout_by_symbol = {c.instrument.symbol: c for c in await BreakoutStrategy().screen(ctx)}

    fusion = next(c for c in result.candidates if c.instrument.symbol == "FUSION")
    solo_best = max(momentum_by_symbol["FUSION"].score, breakout_by_symbol["FUSION"].score)
    expected = min(100.0, solo_best + 5.0)  # one strategy beyond the first -> single 5-point bonus
    assert fusion.score == pytest.approx(expected, rel=1e-6)
    assert fusion.score > solo_best


async def test_persist_screen_result_round_trips(tmp_path: Path) -> None:
    settings = AppSettings(data_dir=tmp_path, _env_file=None)  # type: ignore[call-arg]
    await init_db(settings)

    with BarStore(tmp_path / "bars.duckdb") as store:
        instruments = await _seed_store(store)
        universe_provider = StaticUniverseProvider({US_NASDAQ.code: instruments})
        result = await run_screen(
            US_NASDAQ, store=store, universe_provider=universe_provider, top_n=5
        )

    run_id = await persist_screen_result(result, settings)
    assert run_id > 0

    async with async_session(settings) as session:
        run = (
            await session.execute(select(ScreenRun).where(ScreenRun.id == run_id))
        ).scalar_one()
        assert run.market == US_NASDAQ.code
        assert run.universe_size == 6
        assert run.status == "completed"
        assert sorted(run.strategies_json["strategies"]) == ["breakout", "momentum"]

        picks = (
            await session.execute(select(DailyPick).where(DailyPick.run_id == run_id))
        ).scalars().all()

    assert len(picks) == len(result.candidates)
    by_symbol = {p.symbol: p for p in picks}
    assert set(by_symbol) == {"FUSION", "MOMO"}

    fusion_pick = by_symbol["FUSION"]
    assert "momentum" in fusion_pick.strategy
    assert "breakout" in fusion_pick.strategy
    assert fusion_pick.entry is not None
    assert fusion_pick.stop is not None and fusion_pick.stop < fusion_pick.entry
    assert fusion_pick.target is not None and fusion_pick.target > fusion_pick.entry
    assert isinstance(fusion_pick.features_json, dict)
    assert "close" in fusion_pick.features_json
    assert fusion_pick.llm_verdict_json is None
