"""OrderflowConfluenceStrategy -- long candidates backed by >=2 independent
OHLCV-derived orderflow signals (Task 13).

Every signal here rides on the approximations documented in
``argus.orderflow`` -- there is no tick/L2 data behind any of it. Checked
per instrument, using ``ctx.orderflow()``/``ctx.features()`` plus a direct
sweep/key-level check over raw bars (see ``_recent_reclaimed_low_sweep`` --
``ctx.orderflow()`` only evaluates *today's* bar, but "sweep in the last 3
sessions" needs to look back further):

  (a) a low sweep that reclaimed within the last 3 sessions
  (b) close above the volume-profile POC, within (0%, 8%] of it
  (c) an unfilled gap up with rvol >= 1.5
  (d) close reclaimed the prior week's high

A candidate needs at least 2 of the 4 to survive the gate.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from argus.markets import Instrument
from argus.orderflow.features import OrderflowFeatures
from argus.orderflow.liquidity import SweepSignal, detect_sweeps, key_levels
from argus.screener.base import Candidate, ScreenContext, Strategy
from argus.screener.registry import register_strategy
from argus.screener.scoring import percentile_rank

_SWEEP_LOOKBACK_SESSIONS = 3
_POC_PROXIMITY_MAX_PCT = 8.0
_MIN_GAP_RVOL = 1.5
_MIN_CONFLUENCES = 2
_ATR_STOP_BUFFER_MULT = 0.5
_TARGET_RR = 2.5
_CONFLUENCE_WEIGHT = 0.6
_RVOL_WEIGHT = 0.4


def _recent_reclaimed_low_sweep(
    bars: NDArray[np.void], sessions: int = _SWEEP_LOOKBACK_SESSIONS
) -> SweepSignal | None:
    """The most recent reclaimed low sweep within the last ``sessions``
    sessions, or ``None``. ``detect_sweeps`` only ever looks at the *last*
    bar of whatever's passed to it, so each session back is checked by
    truncating ``bars`` to end there in turn."""
    n = len(bars)
    for back in range(min(sessions, max(n - 1, 0))):
        end = n - back
        window = bars[:end]
        if len(window) < 2:
            break
        for signal in detect_sweeps(window):
            if signal.kind == "low_sweep" and signal.reclaimed:
                return signal
    return None


def _evaluate_confluences(
    bars: NDArray[np.void], of: OrderflowFeatures, features: dict[str, float], close: float
) -> tuple[list[str], SweepSignal | None]:
    matches: list[str] = []

    recent_sweep = _recent_reclaimed_low_sweep(bars)
    if recent_sweep is not None:
        matches.append(
            f"low sweep @ {recent_sweep.level:.2f} reclaimed within "
            f"{_SWEEP_LOOKBACK_SESSIONS}d"
        )

    if (
        of.vp_poc is not None
        and of.close_vs_poc_pct is not None
        and 0.0 < of.close_vs_poc_pct <= _POC_PROXIMITY_MAX_PCT
    ):
        matches.append(f"close {of.close_vs_poc_pct:+.1f}% above VP POC {of.vp_poc:.2f}")

    rvol = features.get("rvol", float("nan"))
    if (
        of.gap_kind == "gap_up"
        and of.gap_filled is False
        and not math.isnan(rvol)
        and rvol >= _MIN_GAP_RVOL
    ):
        gap_pct_label = f"{of.gap_pct:.1f}%" if of.gap_pct is not None else "?"
        matches.append(f"unfilled gap up {gap_pct_label}, rvol {rvol:.2f}x")

    prior_week_high = key_levels(bars).get("prior_week_high")
    if prior_week_high is not None and close > prior_week_high:
        matches.append(f"reclaimed prior week high {prior_week_high:.2f}")

    return matches, recent_sweep


@register_strategy
class OrderflowConfluenceStrategy(Strategy):
    slug = "orderflow_confluence"
    name = "Orderflow Confluence"

    async def screen(self, ctx: ScreenContext) -> list[Candidate]:
        instruments = await ctx.universe()

        evaluated: list[tuple[Instrument, dict[str, float], list[str], float, float, float]] = []
        for inst in instruments:
            features = await ctx.features(inst)
            close = features.get("close", float("nan"))
            atr_14 = features.get("atr_14", float("nan"))
            if math.isnan(close) or math.isnan(atr_14) or atr_14 <= 0:
                continue

            of = await ctx.orderflow(inst)
            if of is None:
                continue

            bars = await ctx.bars(inst)
            matches, recent_sweep = _evaluate_confluences(bars, of, features, close)
            if len(matches) < _MIN_CONFLUENCES:
                continue

            value_area_low = of.vp_val if of.vp_val is not None else close
            sweep_level = recent_sweep.level if recent_sweep is not None else value_area_low
            stop = min(sweep_level, value_area_low) - _ATR_STOP_BUFFER_MULT * atr_14
            if stop >= close:
                continue  # degenerate bracket -- skip rather than emit a nonsensical stop
            target = close + _TARGET_RR * (close - stop)

            rvol_raw = features.get("rvol", float("nan"))
            rvol_for_rank = 0.0 if math.isnan(rvol_raw) else rvol_raw

            evaluated.append((inst, features, matches, stop, target, rvol_for_rank))

        if not evaluated:
            return []

        confluence_counts = [float(len(m)) for _, _, m, _, _, _ in evaluated]
        rvol_values = [r for _, _, _, _, _, r in evaluated]

        candidates: list[Candidate] = []
        for inst, features, matches, stop, target, rvol_for_rank in evaluated:
            confluence_rank = percentile_rank(confluence_counts, float(len(matches)))
            rvol_rank = percentile_rank(rvol_values, rvol_for_rank)
            score = _CONFLUENCE_WEIGHT * confluence_rank + _RVOL_WEIGHT * rvol_rank

            close = features["close"]
            reason = f"{len(matches)} orderflow confluences: " + "; ".join(matches)

            candidates.append(
                Candidate(
                    instrument=inst,
                    strategy=self.slug,
                    score=score,
                    direction="long",
                    stage="orderflow",
                    reason=reason,
                    entry=close,
                    stop=stop,
                    target=target,
                    features=features,
                )
            )
        return candidates
