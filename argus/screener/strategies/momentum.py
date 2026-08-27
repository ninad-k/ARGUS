"""MomentumStrategy — relative momentum ranking within the screened universe.

Requires an established, still-healthy uptrend (``close > sma_50 > sma_200``,
``rsi_14 < 80`` so it isn't already blown off), then ranks survivors by a
percentile blend of 6-month and 1-month rate of change. Scores are relative
to the other candidates in *this* run, not an absolute threshold — the same
instrument's score can shift between runs even with unchanged features, if
the rest of the screened universe moves.
"""

from __future__ import annotations

import math

from argus.markets import Instrument
from argus.screener.base import Candidate, ScreenContext, Strategy
from argus.screener.registry import register_strategy
from argus.screener.scoring import percentile_rank

_RSI_BLOWOFF = 80.0
_ATR_STOP_MULT = 2.0
_ATR_TARGET_MULT = 3.0
_ROC_60_WEIGHT = 0.6
_ROC_20_WEIGHT = 0.4

_REQUIRED_FEATURES = ("close", "sma_50", "sma_200", "rsi_14", "roc_20", "roc_60", "atr_14")


@register_strategy
class MomentumStrategy(Strategy):
    slug = "momentum"
    name = "Momentum"

    async def screen(self, ctx: ScreenContext) -> list[Candidate]:
        instruments = await ctx.universe()

        gated: list[tuple[Instrument, dict[str, float]]] = []
        for inst in instruments:
            features = await ctx.features(inst)
            if self._passes_gate(features):
                gated.append((inst, features))

        if not gated:
            return []

        roc_60_values = [f["roc_60"] for _, f in gated]
        roc_20_values = [f["roc_20"] for _, f in gated]

        candidates: list[Candidate] = []
        for inst, features in gated:
            rank_60 = percentile_rank(roc_60_values, features["roc_60"])
            rank_20 = percentile_rank(roc_20_values, features["roc_20"])
            score = _ROC_60_WEIGHT * rank_60 + _ROC_20_WEIGHT * rank_20

            close = features["close"]
            atr_14 = features["atr_14"]
            entry = close
            stop = close - _ATR_STOP_MULT * atr_14
            target = close + _ATR_TARGET_MULT * atr_14

            reason = (
                f"roc_60 {features['roc_60']:.1f}% ({rank_60:.0f}th pct of screened universe), "
                f"roc_20 {features['roc_20']:.1f}% ({rank_20:.0f}th pct), "
                f"close above rising 50/200 SMA, rsi {features['rsi_14']:.0f} "
                f"(< {_RSI_BLOWOFF:.0f}, not blown off)"
            )

            candidates.append(
                Candidate(
                    instrument=inst,
                    strategy=self.slug,
                    score=score,
                    direction="long",
                    stage="uptrend",
                    reason=reason,
                    entry=entry,
                    stop=stop,
                    target=target,
                    features=features,
                )
            )
        return candidates

    @staticmethod
    def _passes_gate(features: dict[str, float]) -> bool:
        if any(math.isnan(features.get(k, float("nan"))) for k in _REQUIRED_FEATURES):
            return False
        if features["atr_14"] <= 0:
            return False  # degenerate/flat history -- stop/target math needs positive ATR
        return (
            features["close"] > features["sma_50"] > features["sma_200"]
            and features["rsi_14"] < _RSI_BLOWOFF
        )
