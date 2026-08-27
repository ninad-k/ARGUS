"""Volume-profile histogram math on a hand-computed 3-bar fixture, plus
``n_bins`` edge cases and the empty-input ``None`` path."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
from numpy.typing import NDArray

from argus.data.prices.base import bars_from_columns
from argus.orderflow.volume_profile import volume_profile

_START = date(2026, 1, 2)


def _bars(rows: list[tuple[float, float, float]]) -> NDArray[np.void]:
    """Build a bars array from ``(low, high, volume)`` triples -- open/close
    don't matter for volume_profile, set to the bar's midpoint."""
    n = len(rows)
    ts = np.array(
        [np.datetime64((_START + timedelta(days=i)).isoformat(), "s") for i in range(n)],
        dtype="datetime64[s]",
    )
    lows = np.array([r[0] for r in rows], dtype=np.float64)
    highs = np.array([r[1] for r in rows], dtype=np.float64)
    volumes = np.array([r[2] for r in rows], dtype=np.float64)
    mids = (lows + highs) / 2.0
    return bars_from_columns(ts, mids, highs, lows, mids, volumes)


def test_volume_profile_hand_computed_three_bars() -> None:
    # Three bars, each exactly spanning one of three equal-width bins over
    # [0, 30]: bin0=[0,10), bin1=[10,20), bin2=[20,30]. Volumes 100/300/100
    # (total 500) put the POC squarely in bin1 (mid price 15).
    bars = _bars([(0.0, 10.0, 100.0), (10.0, 20.0, 300.0), (20.0, 30.0, 100.0)])

    vp = volume_profile(bars, n_bins=3)

    assert vp is not None
    assert vp.poc == 15.0
    assert np.isclose(np.sum(vp.bins), 500.0)
    assert np.isclose(vp.bins[0], 100.0)
    assert np.isclose(vp.bins[1], 300.0)
    assert np.isclose(vp.bins[2], 100.0)

    # Value area: start at bin1 (300, 60% of 500 -- short of the 70% target),
    # grow toward whichever neighbor carries more volume. Both neighbors tie
    # at 100 -- ties resolve toward the bin above (see _value_area_range) --
    # so the value area becomes bins[1:3] = [10, 30], covering 400/500 = 80%.
    assert vp.value_area_low == 10.0
    assert vp.value_area_high == 30.0


def test_volume_profile_single_bin_covers_full_range() -> None:
    bars = _bars([(0.0, 10.0, 50.0), (5.0, 15.0, 50.0)])

    vp = volume_profile(bars, n_bins=1)

    assert vp is not None
    assert vp.poc == 7.5  # midpoint of [0, 15]
    assert vp.value_area_low == 0.0
    assert vp.value_area_high == 15.0
    assert len(vp.bins) == 1
    assert np.isclose(vp.bins[0], 100.0)


def test_volume_profile_zero_bins_returns_none() -> None:
    bars = _bars([(0.0, 10.0, 100.0)])
    assert volume_profile(bars, n_bins=0) is None


def test_volume_profile_negative_bins_returns_none() -> None:
    bars = _bars([(0.0, 10.0, 100.0)])
    assert volume_profile(bars, n_bins=-1) is None


def test_volume_profile_empty_bars_returns_none() -> None:
    empty = _bars([])
    assert volume_profile(empty) is None


def test_volume_profile_zero_total_volume_returns_none() -> None:
    bars = _bars([(0.0, 10.0, 0.0), (5.0, 15.0, 0.0)])
    assert volume_profile(bars) is None


def test_volume_profile_degenerate_price_range_returns_none() -> None:
    # Every bar has identical high == low -- zero price span overall.
    bars = _bars([(10.0, 10.0, 100.0), (10.0, 10.0, 100.0)])
    assert volume_profile(bars) is None
