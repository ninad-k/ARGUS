"""Liquidity-sweep detection and key price levels, from daily bars only.

A "sweep" here is an OHLCV-only proxy for a liquidity grab: a session trades
beyond a prior N-day high/low (a level where resting stop orders are commonly
assumed to cluster) and the close then tells whether that move was rejected
(``reclaimed=True`` -- the close came back inside the prior range, the
classic "stop hunt then reverse" pattern) or accepted (``reclaimed=False`` --
the close held beyond the level, i.e. a genuine breakout/breakdown rather
than a sweep). This is inferred from price action alone; there is no L2/tape
evidence of an actual resting-order sweep behind it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

_DEFAULT_LOOKBACK = 20
_WEEK_SESSIONS = 5
_MONTH_SESSIONS = 20


@dataclass(frozen=True)
class SweepSignal:
    """One sweep beyond a prior high/low, evaluated for the *last* bar in
    whatever ``daily_bars`` array was passed to ``detect_sweeps`` -- callers
    checking whether a sweep happened N sessions ago pass a bars array
    truncated to end N sessions back (see e.g.
    ``argus.screener.strategies.orderflow_confluence``)."""

    kind: Literal["high_sweep", "low_sweep"]
    level: float
    reclaimed: bool


def detect_sweeps(
    daily_bars: NDArray[np.void], lookback: int = _DEFAULT_LOOKBACK
) -> list[SweepSignal]:
    """Sweep signals for the last bar in ``daily_bars`` against the prior
    ``lookback`` sessions' high/low (excluding the last bar itself).

    Returns 0, 1 (a high-only or low-only sweep), or 2 (both -- an outside
    bar that traded beyond both the prior high and low) signals. Empty when
    there's fewer than 2 bars, or the last bar didn't trade beyond either
    level.
    """
    n = len(daily_bars)
    if n < 2:
        return []

    window = daily_bars[-(lookback + 1) : -1]
    if len(window) == 0:
        return []

    today = daily_bars[-1]
    prior_high = float(np.max(window["high"]))
    prior_low = float(np.min(window["low"]))

    today_high = float(today["high"])
    today_low = float(today["low"])
    today_close = float(today["close"])

    signals: list[SweepSignal] = []
    if today_high > prior_high:
        signals.append(
            SweepSignal(kind="high_sweep", level=prior_high, reclaimed=today_close < prior_high)
        )
    if today_low < prior_low:
        signals.append(
            SweepSignal(kind="low_sweep", level=prior_low, reclaimed=today_close > prior_low)
        )
    return signals


def key_levels(daily_bars: NDArray[np.void]) -> dict[str, float]:
    """Prior-session/week/20-session high-low reference levels, keyed off
    the *last* bar in ``daily_bars`` as "today". "Week"/"20d" here mean
    trailing trading sessions (5 / 20), not calendar-week boundaries -- daily
    bars carry no calendar-week grouping of their own.

    A level whose window doesn't fit in the available history is simply
    omitted from the returned dict (not filled with NaN) -- callers should
    use ``.get()``.
    """
    n = len(daily_bars)
    out: dict[str, float] = {}
    if n >= 2:
        prior_day = daily_bars[-2]
        out["prior_day_high"] = float(prior_day["high"])
        out["prior_day_low"] = float(prior_day["low"])
    if n >= _WEEK_SESSIONS + 1:
        week_window = daily_bars[-(_WEEK_SESSIONS + 1) : -1]
        out["prior_week_high"] = float(np.max(week_window["high"]))
        out["prior_week_low"] = float(np.min(week_window["low"]))
    if n >= _MONTH_SESSIONS + 1:
        window_20d = daily_bars[-(_MONTH_SESSIONS + 1) : -1]
        out["20d_high"] = float(np.max(window_20d["high"]))
        out["20d_low"] = float(np.min(window_20d["low"]))
    return out
