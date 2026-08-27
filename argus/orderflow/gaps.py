"""Gap classification from daily bars.

Compares the last bar's open to the prior bar's close. OHLCV-only: there's no
way to distinguish a genuine liquidity/news gap from e.g. a dividend or split
adjustment artifact without corporate-actions data, which isn't consulted
here -- a caller feeding un-adjusted bars around an ex-dividend date will see
a spurious small gap.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

_GAP_THRESHOLD_PCT = 0.5


def classify_gap(daily_bars: NDArray[np.void]) -> dict[str, float | str | bool] | None:
    """Gap percentage/kind/fill-status for the last bar in ``daily_bars``
    relative to the prior bar's close.

    ``kind`` is ``"gap_up"``/``"gap_down"`` when ``|gap_pct| >= 0.5%``, else
    ``"none"``. ``filled`` is whether the gapping session's own
    ``[low, high]`` range re-touched the prior close (i.e. the gap was at
    least partially "filled" intraday).

    ``None`` when there's fewer than two bars to compare, or the prior
    close is zero (degenerate/bad data -- gap % is undefined).
    """
    n = len(daily_bars)
    if n < 2:
        return None

    prev_close = float(daily_bars[-2]["close"])
    if prev_close == 0:
        return None

    today = daily_bars[-1]
    today_open = float(today["open"])
    today_high = float(today["high"])
    today_low = float(today["low"])

    gap_pct = (today_open - prev_close) / prev_close * 100.0

    if gap_pct >= _GAP_THRESHOLD_PCT:
        kind = "gap_up"
    elif gap_pct <= -_GAP_THRESHOLD_PCT:
        kind = "gap_down"
    else:
        kind = "none"

    filled = today_low <= prev_close <= today_high

    return {"gap_pct": gap_pct, "kind": kind, "filled": filled}
