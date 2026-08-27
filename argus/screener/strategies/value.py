"""ValueStrategy — fundamentals-driven cheapness/quality/growth ranking.

Gates on fundamentals rather than price action: a large-enough business
(``market_cap`` above a market-appropriate floor), trading at a sane multiple
(``0 < pe < 40``), earning a real return on equity (``roe > 10%``), and not
buried in debt (``debt_to_equity < 2.0``, skipped if unknown -- many
providers simply don't report it). A technical "value trap" guard on top of
the fundamentals gate excludes names in a confirmed technical downtrend --
cheap-and-still-falling is not what this strategy is looking for.

Survivors are ranked by a blended percentile score across three axes --
cheapness (lower PE/PB), quality (ROE, profit margin), growth (revenue/
earnings growth) -- weighted 0.40/0.35/0.25. A sub-metric that's ``None`` for
a given name (PB, profit margin, either growth figure) contributes a neutral
50th-percentile rank rather than dropping out of the blend entirely, so one
missing data point doesn't zero out an otherwise-strong candidate.
"""

from __future__ import annotations

import math

from argus.data.fundamentals import FundamentalsView
from argus.markets import Instrument, Market
from argus.screener.base import Candidate, ScreenContext, Strategy
from argus.screener.registry import register_strategy
from argus.screener.scoring import percentile_rank

_US_MIN_MARKET_CAP = 2_000_000_000.0  # $2B
_IN_MIN_MARKET_CAP = 100_000_000_000.0  # ₹100B
_MAX_PE = 40.0
_MIN_ROE = 0.10
_MAX_DEBT_TO_EQUITY = 2.0
_DIST_SMA200_FLOOR_PCT = -10.0  # value-trap guard: allow a shallow break of sma200

_ATR_STOP_MULT = 2.5
_TARGET_MULT = 1.20  # valuation-convergence target, not a technical level

_WEIGHT_CHEAPNESS = 0.40
_WEIGHT_QUALITY = 0.35
_WEIGHT_GROWTH = 0.25

_MEDIAN_RANK = 50.0  # neutral rank assigned to a missing sub-metric

_REQUIRED_FEATURES = ("close", "sma_200", "dist_sma200_pct", "atr_14")


def _min_market_cap(market: Market) -> float:
    return _IN_MIN_MARKET_CAP if market.currency == "INR" else _US_MIN_MARKET_CAP


def _passes_fundamentals_gate(market: Market, fv: FundamentalsView) -> bool:
    if fv.market_cap is None or fv.market_cap < _min_market_cap(market):
        return False
    if fv.pe is None or not (0 < fv.pe < _MAX_PE):
        return False
    if fv.roe is None or fv.roe <= _MIN_ROE:
        return False
    return fv.debt_to_equity is None or fv.debt_to_equity < _MAX_DEBT_TO_EQUITY


def _passes_technical_gate(features: dict[str, float]) -> bool:
    if any(math.isnan(features.get(k, float("nan"))) for k in _REQUIRED_FEATURES):
        return False
    if features["atr_14"] <= 0:
        return False
    # Value-trap guard: either still above sma200, or only a shallow break of it.
    above_sma200 = features["close"] > features["sma_200"]
    shallow_break = features["dist_sma200_pct"] > _DIST_SMA200_FLOOR_PCT
    return above_sma200 or shallow_break


def _rank_or_median(values: list[float], value: float | None) -> float:
    if value is None:
        return _MEDIAN_RANK
    return percentile_rank(values, value)


@register_strategy
class ValueStrategy(Strategy):
    slug = "value"
    name = "Value"

    async def screen(self, ctx: ScreenContext) -> list[Candidate]:
        instruments = await ctx.universe()

        gated: list[tuple[Instrument, dict[str, float], FundamentalsView]] = []
        for inst in instruments:
            fv = await ctx.fundamentals(inst)
            if fv is None or not _passes_fundamentals_gate(ctx.market, fv):
                continue
            features = await ctx.features(inst)
            if not _passes_technical_gate(features):
                continue
            gated.append((inst, features, fv))

        if not gated:
            return []

        pe_values = [fv.pe for _, _, fv in gated if fv.pe is not None]
        pb_values = [fv.pb for _, _, fv in gated if fv.pb is not None]
        roe_values = [fv.roe for _, _, fv in gated if fv.roe is not None]
        margin_values = [fv.profit_margin for _, _, fv in gated if fv.profit_margin is not None]
        rev_growth_values = [
            fv.revenue_growth for _, _, fv in gated if fv.revenue_growth is not None
        ]
        earn_growth_values = [
            fv.earnings_growth for _, _, fv in gated if fv.earnings_growth is not None
        ]

        candidates: list[Candidate] = []
        for inst, features, fv in gated:
            assert fv.pe is not None  # guaranteed by _passes_fundamentals_gate
            assert fv.roe is not None  # guaranteed by _passes_fundamentals_gate

            pe_rank = percentile_rank(pe_values, fv.pe)
            pb_rank = _rank_or_median(pb_values, fv.pb)
            cheapness = ((100.0 - pe_rank) + (100.0 - pb_rank)) / 2.0

            roe_rank = percentile_rank(roe_values, fv.roe)
            margin_rank = _rank_or_median(margin_values, fv.profit_margin)
            quality = (roe_rank + margin_rank) / 2.0

            rev_rank = _rank_or_median(rev_growth_values, fv.revenue_growth)
            earn_rank = _rank_or_median(earn_growth_values, fv.earnings_growth)
            growth = (rev_rank + earn_rank) / 2.0

            score = (
                _WEIGHT_CHEAPNESS * cheapness + _WEIGHT_QUALITY * quality + _WEIGHT_GROWTH * growth
            )
            score = max(0.0, min(100.0, score))

            close = features["close"]
            atr_14 = features["atr_14"]
            entry = close
            stop = close - _ATR_STOP_MULT * atr_14
            target = close * _TARGET_MULT

            reason_bits = [
                f"PE {fv.pe:.1f} ({100.0 - pe_rank:.0f}th pct cheap vs peers)",
                f"ROE {fv.roe * 100:.1f}% ({roe_rank:.0f}th pct)",
            ]
            if fv.revenue_growth is not None:
                reason_bits.append(f"revenue growth {fv.revenue_growth * 100:.1f}%")
            if fv.earnings_growth is not None:
                reason_bits.append(f"earnings growth {fv.earnings_growth * 100:.1f}%")
            if fv.debt_to_equity is not None:
                reason_bits.append(f"debt/equity {fv.debt_to_equity:.2f}")
            reason = ", ".join(reason_bits)

            candidates.append(
                Candidate(
                    instrument=inst,
                    strategy=self.slug,
                    score=score,
                    direction="long",
                    stage="value",
                    reason=reason,
                    entry=entry,
                    stop=stop,
                    target=target,
                    features=features,
                )
            )
        return candidates
