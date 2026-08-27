"""Gap up/down/none classification at the 0.5% threshold boundary, plus
the filled/unfilled and insufficient-history cases."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest
from numpy.typing import NDArray

from argus.data.prices.base import bars_from_columns
from argus.orderflow.gaps import classify_gap

_START = date(2026, 1, 2)


def _bars(rows: list[tuple[float, float, float, float]]) -> NDArray[np.void]:
    """Build a 2-bar array from ``(open, high, low, close)`` rows; volume is
    irrelevant to gap classification, set to a constant."""
    n = len(rows)
    ts = np.array(
        [np.datetime64((_START + timedelta(days=i)).isoformat(), "s") for i in range(n)],
        dtype="datetime64[s]",
    )
    opens = np.array([r[0] for r in rows], dtype=np.float64)
    highs = np.array([r[1] for r in rows], dtype=np.float64)
    lows = np.array([r[2] for r in rows], dtype=np.float64)
    closes = np.array([r[3] for r in rows], dtype=np.float64)
    volumes = np.full(n, 1_000.0)
    return bars_from_columns(ts, opens, highs, lows, closes, volumes)


def test_gap_up_above_threshold() -> None:
    # Prior close 100, today opens at 101 (+1.0%, above the 0.5% threshold).
    bars = _bars([(99.0, 100.5, 98.5, 100.0), (101.0, 102.0, 100.5, 101.5)])
    gap = classify_gap(bars)
    assert gap is not None
    assert gap["kind"] == "gap_up"
    assert gap["gap_pct"] == 1.0


def test_gap_down_below_threshold() -> None:
    bars = _bars([(99.0, 100.5, 98.5, 100.0), (99.0, 99.5, 97.0, 98.0)])
    gap = classify_gap(bars)
    assert gap is not None
    assert gap["kind"] == "gap_down"
    assert gap["gap_pct"] == -1.0


def test_gap_none_within_threshold() -> None:
    # +0.3% open -- inside the 0.5% band, classified as no gap.
    bars = _bars([(99.0, 100.5, 98.5, 100.0), (100.3, 101.0, 100.0, 100.8)])
    gap = classify_gap(bars)
    assert gap is not None
    assert gap["kind"] == "none"
    assert gap["gap_pct"] == pytest.approx(0.3)


def test_gap_up_exactly_at_threshold_boundary() -> None:
    # Exactly +0.5% -- the boundary is inclusive ("gap_pct >= threshold").
    bars = _bars([(100.0, 100.5, 99.5, 100.0), (100.5, 101.0, 100.0, 100.8)])
    gap = classify_gap(bars)
    assert gap is not None
    assert gap["kind"] == "gap_up"
    assert gap["gap_pct"] == 0.5


def test_gap_down_exactly_at_threshold_boundary() -> None:
    bars = _bars([(100.0, 100.5, 99.5, 100.0), (99.5, 100.0, 99.0, 99.2)])
    gap = classify_gap(bars)
    assert gap is not None
    assert gap["kind"] == "gap_down"
    assert gap["gap_pct"] == -0.5


def test_gap_up_filled_when_range_retouches_prior_close() -> None:
    # Prior close 100, today gaps up to 102 but trades back down to 99.5
    # intraday -- the gap is filled.
    bars = _bars([(99.0, 100.5, 98.5, 100.0), (102.0, 102.5, 99.5, 101.0)])
    gap = classify_gap(bars)
    assert gap is not None
    assert gap["filled"] is True


def test_gap_up_unfilled_when_range_never_retouches_prior_close() -> None:
    # Today's low (101.0) never comes back down to prior close (100.0).
    bars = _bars([(99.0, 100.5, 98.5, 100.0), (102.0, 103.0, 101.0, 102.5)])
    gap = classify_gap(bars)
    assert gap is not None
    assert gap["filled"] is False


def test_classify_gap_insufficient_history_returns_none() -> None:
    bars = _bars([(100.0, 101.0, 99.0, 100.0)])
    assert classify_gap(bars) is None


def test_classify_gap_zero_prior_close_returns_none() -> None:
    bars = _bars([(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0)])
    assert classify_gap(bars) is None
