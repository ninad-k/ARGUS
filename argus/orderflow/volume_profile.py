"""Volume profile -- a histogram of traded volume by price.

A real volume profile is built from the tape: which price each individual
print traded at. ARGUS only has OHLCV bars, so this is the standard
OHLCV-only approximation instead: each bar's *total* volume is distributed
*uniformly* across its own ``[low, high]`` range (i.e. assumed to trade
evenly at every price the bar touched), then accumulated into fixed-width
price bins spanning the whole bars array's range. This is a coarse proxy --
a bar that actually printed most of its volume at the open, or in one sharp
spike, looks identical here to one that traded evenly throughout -- not a
substitute for tape/L2-based volume-at-price.

Works on whatever bars are passed in: intraday bars give a finer, more
realistic profile (more, smaller bars -> less within-bar smearing); daily
bars are a coarser fallback when no intraday history is available.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

_DEFAULT_BINS = 30
_VALUE_AREA_PCT = 0.70


@dataclass(frozen=True)
class VolumeProfile:
    """One volume-profile snapshot. ``bins``/``bin_edges`` are the full
    histogram (``len(bin_edges) == len(bins) + 1``) for callers that want
    more than just the summary POC/value-area numbers (e.g. a UI chart)."""

    poc: float
    value_area_low: float
    value_area_high: float
    bins: NDArray[np.float64]
    bin_edges: NDArray[np.float64]


def volume_profile(bars: NDArray[np.void], n_bins: int = _DEFAULT_BINS) -> VolumeProfile | None:
    """Build a ``VolumeProfile`` from ``bars`` (any BAR_DTYPE array, ascending
    or not -- order doesn't matter here). ``None`` when there's nothing to
    build one from: an empty ``bars`` array, ``n_bins < 1``, a degenerate
    (zero-width) price range, or zero total volume.

    POC ("point of control") is the price at the center of the highest-volume
    bin. The value area is the smallest contiguous run of bins around the POC
    whose volume sums to >= 70% of total volume, grown one bin at a time
    toward whichever side (above/below the current range) carries more
    volume -- the standard value-area construction.
    """
    if len(bars) == 0 or n_bins < 1:
        return None

    high = bars["high"].astype(np.float64)
    low = bars["low"].astype(np.float64)
    volume = bars["volume"].astype(np.float64)

    lo = float(np.min(low))
    hi = float(np.max(high))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None

    bin_edges = np.linspace(lo, hi, n_bins + 1)
    bins = np.zeros(n_bins, dtype=np.float64)
    span = hi - lo

    for bar_high, bar_low, bar_volume in zip(high, low, volume, strict=True):
        if bar_volume <= 0:
            continue
        bar_lo = max(bar_low, lo)
        bar_hi = min(bar_high, hi)
        if bar_hi <= bar_lo:
            # Degenerate bar range (high == low, or entirely out of [lo, hi]
            # due to float rounding at the extremes) -- drop the whole
            # volume into the single bin containing that price.
            idx = min(int((bar_lo - lo) / span * n_bins), n_bins - 1)
            bins[idx] += bar_volume
            continue

        bar_range = bar_hi - bar_lo
        lo_idx = min(int((bar_lo - lo) / span * n_bins), n_bins - 1)
        hi_idx = min(int((bar_hi - lo) / span * n_bins), n_bins - 1)
        for idx in range(lo_idx, hi_idx + 1):
            edge_lo = max(bin_edges[idx], bar_lo)
            edge_hi = min(bin_edges[idx + 1], bar_hi)
            overlap = max(0.0, edge_hi - edge_lo)
            if overlap > 0:
                bins[idx] += bar_volume * (overlap / bar_range)

    total = float(np.sum(bins))
    if total <= 0:
        return None

    poc_idx = int(np.argmax(bins))
    poc = float((bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2.0)

    lo_idx, hi_idx = _value_area_range(bins, poc_idx, total)
    value_area_low = float(bin_edges[lo_idx])
    value_area_high = float(bin_edges[hi_idx + 1])

    return VolumeProfile(
        poc=poc,
        value_area_low=value_area_low,
        value_area_high=value_area_high,
        bins=bins,
        bin_edges=bin_edges,
    )


def _value_area_range(
    bins: NDArray[np.float64], poc_idx: int, total: float
) -> tuple[int, int]:
    """Smallest contiguous ``[lo_idx, hi_idx]`` bin range around ``poc_idx``
    covering >= ``_VALUE_AREA_PCT`` of ``total`` volume, grown one bin at a
    time toward whichever side currently carries more volume."""
    lo_idx = hi_idx = poc_idx
    covered = float(bins[poc_idx])
    target = total * _VALUE_AREA_PCT
    n = len(bins)
    while covered < target and (lo_idx > 0 or hi_idx < n - 1):
        below = bins[lo_idx - 1] if lo_idx > 0 else -1.0
        above = bins[hi_idx + 1] if hi_idx < n - 1 else -1.0
        if above >= below:
            hi_idx += 1
            covered += float(bins[hi_idx])
        else:
            lo_idx -= 1
            covered += float(bins[lo_idx])
    return lo_idx, hi_idx
