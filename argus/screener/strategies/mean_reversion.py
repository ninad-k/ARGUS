"""MeanReversionStrategy — long-only pullback-in-uptrend.

Looks for names that are still in a primary uptrend (``close > sma_200``,
``roc_60 > 0`` so the longer-term trend hasn't actually broken) but have
pulled back into oversold territory on the daily timeframe (``rsi_14 < 35``,
``close < sma_20``), with volume not having collapsed (``rvol > 0.5`` -- a
totally dried-up tape makes any reversion signal unreliable, dead-cat or
otherwise).

Score blends how deep the oversold reading is against how strong the
underlying uptrend is (``adx_14`` and distance above ``sma_200``), each
percentile-ranked against the other gate-passing candidates, 50/50 -- so a
shallow dip in a very strong trend and a deep dip in a merely-decent trend
can both land near the top.
"""

from __future__ import annotations

import math

import numpy as np

from argus.markets import Instrument
from argus.screener.base import Candidate, ScreenContext, Strategy
from argus.screener.registry import register_strategy
from argus.screener.scoring import percentile_rank

_RSI_OVERSOLD_MAX = 35.0
_MIN_RVOL = 0.5
_RECENT_LOW_LOOKBACK = 20
_ATR_STOP_MULT = 2.0
_RECENT_LOW_STOP_MULT = 0.5
_ATR_TARGET_FALLBACK_MULT = 2.0

_WEIGHT_OVERSOLD = 0.5
_WEIGHT_TREND = 0.5

_REQUIRED_FEATURES = (
    "close",
    "sma_200",
    "sma_20",
    "rsi_14",
    "roc_60",
    "rvol",
    "atr_14",
    "adx_14",
    "dist_sma200_pct",
)


def _passes_gate(features: dict[str, float]) -> bool:
    if any(math.isnan(features.get(k, float("nan"))) for k in _REQUIRED_FEATURES):
        return False
    if features["atr_14"] <= 0:
        return False
    return (
        features["close"] > features["sma_200"]  # still in a primary uptrend
        and features["rsi_14"] < _RSI_OVERSOLD_MAX  # oversold on the daily
        and features["close"] < features["sma_20"]  # actually pulled back
        and features["roc_60"] > 0  # longer-term momentum intact
        and features["rvol"] > _MIN_RVOL  # tape hasn't dried up
    )


@register_strategy
class MeanReversionStrategy(Strategy):
    slug = "mean_reversion"
    name = "Mean Reversion"

    async def screen(self, ctx: ScreenContext) -> list[Candidate]:
        instruments = await ctx.universe()

        gated: list[tuple[Instrument, dict[str, float]]] = []
        for inst in instruments:
            features = await ctx.features(inst)
            if _passes_gate(features):
                gated.append((inst, features))

        if not gated:
            return []

        rsi_values = [f["rsi_14"] for _, f in gated]
        adx_values = [f["adx_14"] for _, f in gated]
        dist_values = [f["dist_sma200_pct"] for _, f in gated]

        candidates: list[Candidate] = []
        for inst, features in gated:
            # Lower RSI = deeper oversold = better -> invert the rank.
            oversold_rank = 100.0 - percentile_rank(rsi_values, features["rsi_14"])
            adx_rank = percentile_rank(adx_values, features["adx_14"])
            dist_rank = percentile_rank(dist_values, features["dist_sma200_pct"])
            trend_rank = (adx_rank + dist_rank) / 2.0

            score = _WEIGHT_OVERSOLD * oversold_rank + _WEIGHT_TREND * trend_rank
            score = max(0.0, min(100.0, score))

            close = features["close"]
            atr_14 = features["atr_14"]
            sma_20 = features["sma_20"]

            bars = await ctx.bars(inst)
            low = bars["low"].astype(np.float64)
            recent_low = float(np.min(low[-_RECENT_LOW_LOOKBACK:])) if len(low) > 0 else close

            entry = close
            stop = min(
                close - _ATR_STOP_MULT * atr_14,
                recent_low - _RECENT_LOW_STOP_MULT * atr_14,
            )
            # sma_20 is the reversion target; if it's already below entry (can
            # happen right at the gate boundary) fall back to an ATR target.
            target = sma_20 if sma_20 > entry else close + _ATR_TARGET_FALLBACK_MULT * atr_14

            reason = (
                f"rsi {features['rsi_14']:.0f} ({oversold_rank:.0f}th pct oversold), "
                f"{features['dist_sma200_pct']:.1f}% above sma200, "
                f"adx {features['adx_14']:.0f} ({adx_rank:.0f}th pct trend strength), "
                f"roc_60 {features['roc_60']:.1f}% (uptrend intact), rvol {features['rvol']:.2f}x"
            )

            candidates.append(
                Candidate(
                    instrument=inst,
                    strategy=self.slug,
                    score=score,
                    direction="long",
                    stage="pullback",
                    reason=reason,
                    entry=entry,
                    stop=stop,
                    target=target,
                    features=features,
                )
            )
        return candidates
