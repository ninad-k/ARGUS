"""Sweep detection (reclaimed vs. held vs. no sweep) and key_levels values,
on hand-built daily-bar fixtures."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
from numpy.typing import NDArray

from argus.data.prices.base import bars_from_columns
from argus.orderflow.liquidity import detect_sweeps, key_levels

_START = date(2026, 1, 2)


def _bars(rows: list[tuple[float, float, float, float, float]]) -> NDArray[np.void]:
    """Build a bars array from ``(open, high, low, close, volume)`` rows."""
    n = len(rows)
    ts = np.array(
        [np.datetime64((_START + timedelta(days=i)).isoformat(), "s") for i in range(n)],
        dtype="datetime64[s]",
    )
    opens = np.array([r[0] for r in rows], dtype=np.float64)
    highs = np.array([r[1] for r in rows], dtype=np.float64)
    lows = np.array([r[2] for r in rows], dtype=np.float64)
    closes = np.array([r[3] for r in rows], dtype=np.float64)
    volumes = np.array([r[4] for r in rows], dtype=np.float64)
    return bars_from_columns(ts, opens, highs, lows, closes, volumes)


def _flat_history(n: int, price: float = 100.0) -> list[tuple[float, float, float, float, float]]:
    return [(price, price + 1.0, price - 1.0, price, 1_000.0) for _ in range(n)]


def test_low_sweep_reclaimed_when_close_back_inside() -> None:
    # 20 flat sessions with low held at 99, then a bar that trades to 95
    # (beyond the prior 20d low of 99) but closes back at 99.5 -- reclaimed.
    rows = _flat_history(20, price=100.0)
    rows.append((99.5, 100.5, 95.0, 99.5, 5_000.0))
    bars = _bars(rows)

    signals = detect_sweeps(bars)

    assert len(signals) == 1
    signal = signals[0]
    assert signal.kind == "low_sweep"
    assert signal.level == 99.0
    assert signal.reclaimed is True


def test_low_sweep_not_reclaimed_when_close_holds_beyond() -> None:
    # Same setup, but today closes at 96 -- beyond (below) the prior low --
    # a genuine breakdown, not a reclaimed sweep.
    rows = _flat_history(20, price=100.0)
    rows.append((99.5, 100.5, 95.0, 96.0, 5_000.0))
    bars = _bars(rows)

    signals = detect_sweeps(bars)

    assert len(signals) == 1
    signal = signals[0]
    assert signal.kind == "low_sweep"
    assert signal.reclaimed is False


def test_high_sweep_reclaimed_when_close_back_inside() -> None:
    rows = _flat_history(20, price=100.0)
    rows.append((100.5, 105.0, 99.5, 100.5, 5_000.0))
    bars = _bars(rows)

    signals = detect_sweeps(bars)

    assert len(signals) == 1
    signal = signals[0]
    assert signal.kind == "high_sweep"
    assert signal.level == 101.0
    assert signal.reclaimed is True


def test_no_sweep_when_today_stays_inside_prior_range() -> None:
    rows = _flat_history(20, price=100.0)
    rows.append((100.0, 100.5, 99.5, 100.0, 1_000.0))
    bars = _bars(rows)

    assert detect_sweeps(bars) == []


def test_outside_bar_produces_both_sweep_signals() -> None:
    rows = _flat_history(20, price=100.0)
    # Trades beyond both the prior high (101) and low (99), closes flat.
    rows.append((100.0, 105.0, 95.0, 100.0, 5_000.0))
    bars = _bars(rows)

    signals = detect_sweeps(bars)
    kinds = {s.kind for s in signals}

    assert kinds == {"high_sweep", "low_sweep"}


def test_detect_sweeps_insufficient_history_returns_empty() -> None:
    bars = _bars(_flat_history(1))
    assert detect_sweeps(bars) == []


def test_key_levels_prior_day() -> None:
    # bars[-2] (the session before "today") is the reference for prior_day_*
    # -- today's own (much wider) range must not leak into it.
    rows = [(100.0, 101.0, 99.0, 100.0, 1_000.0)]
    rows.append((100.0, 103.0, 98.0, 101.0, 1_000.0))  # today
    bars = _bars(rows)

    levels = key_levels(bars)

    assert levels["prior_day_high"] == 101.0
    assert levels["prior_day_low"] == 99.0
    assert "prior_week_high" not in levels
    assert "20d_high" not in levels


def test_key_levels_prior_week_and_20d() -> None:
    # 20 days of rising highs/lows (day i: high=100+i, low=90+i), plus today.
    rows = [
        (95.0 + i, 100.0 + i, 90.0 + i, 97.0 + i, 1_000.0) for i in range(20)
    ]
    rows.append((120.0, 125.0, 118.0, 122.0, 2_000.0))  # today
    bars = _bars(rows)

    levels = key_levels(bars)

    # prior_week = trailing 5 sessions excluding today -> days index 15..19
    assert levels["prior_week_high"] == 100.0 + 19
    assert levels["prior_week_low"] == 90.0 + 15
    # 20d = trailing 20 sessions excluding today -> days index 0..19
    assert levels["20d_high"] == 100.0 + 19
    assert levels["20d_low"] == 90.0 + 0
    assert levels["prior_day_high"] == 100.0 + 19
    assert levels["prior_day_low"] == 90.0 + 19


def test_key_levels_empty_bars_returns_empty_dict() -> None:
    assert key_levels(_bars([])) == {}
