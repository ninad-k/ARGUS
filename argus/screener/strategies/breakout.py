"""BreakoutStrategy — near-52-week-high consolidation with a volume trigger.

Looks for names within 3% of their 252-day high whose 20-day Bollinger Band
width has already contracted into the lower half of its own trailing 120-bar
range (a classic pre-breakout "coil"), with today's volume already picking up
(``rvol >= 1.2``). ``stage`` distinguishes names that have already sliced
through their prior 252-day high ("breakout") from those still consolidating
just under it ("pre-breakout") -- note this compares against the high
*excluding* today's bar, unlike the ``high_252`` feature (which includes it).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from argus.indicators import numpy_impl as _ni
from argus.markets import Instrument
from argus.screener.base import Candidate, ScreenContext, Strategy
from argus.screener.registry import register_strategy

_PCT_OFF_HIGH_MIN = -3.0  # within 3% of the 252-day high
_MIN_RVOL = 1.2
_BB_HISTORY_BARS = 120
_CONTRACTION_PCT_MAX = 50.0  # lower half of the trailing bb-width range
_ATR_STOP_MULT = 1.5
_TARGET_RR = 2.0

_REQUIRED_FEATURES = ("close", "pct_off_high", "rvol", "adx_14", "atr_14", "high_252")


def _bb_width_percentile(bars: NDArray[np.void]) -> float | None:
    """Percentile rank (0-100, 100 = widest) of the latest 20-day Bollinger Band
    width within its own trailing ``_BB_HISTORY_BARS``-bar history.

    ``None`` if there isn't enough history to judge contraction.
    """
    close = bars["close"].astype(np.float64)
    bb = _ni.bollinger_bands(close, 20, 2.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        width = (bb.arrays["upper"] - bb.arrays["lower"]) / bb.arrays["mid"]
    finite = width[~np.isnan(width)]
    if len(finite) < 2:
        return None
    window = finite[-_BB_HISTORY_BARS:]
    latest = finite[-1]
    return 100.0 * float(np.sum(window <= latest) - 1) / (len(window) - 1)


@register_strategy
class BreakoutStrategy(Strategy):
    slug = "breakout"
    name = "Breakout"

    async def screen(self, ctx: ScreenContext) -> list[Candidate]:
        instruments = await ctx.universe()
        candidates: list[Candidate] = []
        for inst in instruments:
            features = await ctx.features(inst)
            candidate = await self._evaluate(ctx, inst, features)
            if candidate is not None:
                candidates.append(candidate)
        return candidates

    async def _evaluate(
        self, ctx: ScreenContext, inst: Instrument, features: dict[str, float]
    ) -> Candidate | None:
        if any(math.isnan(features.get(k, float("nan"))) for k in _REQUIRED_FEATURES):
            return None
        if features["pct_off_high"] < _PCT_OFF_HIGH_MIN:
            return None
        if features["rvol"] < _MIN_RVOL:
            return None
        if features["atr_14"] <= 0:
            return None

        bars = await ctx.bars(inst)
        contraction_pct = _bb_width_percentile(bars)
        if contraction_pct is None or contraction_pct > _CONTRACTION_PCT_MAX:
            return None  # only trade a coil, not an already-expanded range

        close = features["close"]
        high = bars["high"].astype(np.float64)
        prior_high = float(np.max(high[:-1])) if len(high) > 1 else float("nan")
        stage = "breakout" if not math.isnan(prior_high) and close >= prior_high else "pre-breakout"

        # pct_off_high in [-3, 0] -> 0 at -3%, 100 at the high itself.
        proximity_score = max(
            0.0, 100.0 + features["pct_off_high"] * (100.0 / abs(_PCT_OFF_HIGH_MIN))
        )
        volume_score = min(100.0, max(0.0, (features["rvol"] - 1.0) * 100.0))
        adx_score = min(100.0, max(0.0, features["adx_14"]))
        score = max(0.0, min(100.0, 0.5 * proximity_score + 0.3 * volume_score + 0.2 * adx_score))

        entry = features["high_252"] * 1.001
        stop = close - _ATR_STOP_MULT * features["atr_14"]
        target = entry + _TARGET_RR * (entry - stop)

        reason = (
            f"within {abs(features['pct_off_high']):.1f}% of the 252d high, "
            f"bb width at {contraction_pct:.0f}th pct of its own {_BB_HISTORY_BARS}d range "
            f"(contracted), rvol {features['rvol']:.2f}x, adx {features['adx_14']:.0f}"
        )

        return Candidate(
            instrument=inst,
            strategy=self.slug,
            score=score,
            direction="long",
            stage=stage,
            reason=reason,
            entry=entry,
            stop=stop,
            target=target,
            features=features,
        )
