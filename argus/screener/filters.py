"""Composable pre-screen filters and the default filter chain.

Mirrors DRUVA's ``FilterPipeline``: each filter checks one instrument's
features and returns a rejection reason (or ``None`` to pass). ``FilterChain``
runs filters in order and short-circuits at the first rejection, which keeps
the rejection reasons attributable to a single cause per symbol.
"""

from __future__ import annotations

import math
from typing import Protocol

from argus.markets import Instrument, Market


class InstrumentFilter(Protocol):
    """A single pass/reject check against one instrument's features."""

    def check(self, inst: Instrument, features: dict[str, float]) -> str | None:
        """Return ``None`` if ``inst`` passes, or a human-readable rejection reason."""
        ...


class MinPriceFilter:
    """Rejects instruments trading below ``min_price`` (uses the ``close`` feature)."""

    def __init__(self, min_price: float) -> None:
        self.min_price = min_price

    def check(self, inst: Instrument, features: dict[str, float]) -> str | None:
        close = features.get("close", float("nan"))
        if math.isnan(close):
            return "price unavailable"
        if close < self.min_price:
            return f"price {close:.2f} below minimum {self.min_price:.2f}"
        return None


class MinVolumeFilter:
    """Rejects instruments whose 20-day average volume is below ``min_avg_volume``."""

    def __init__(self, min_avg_volume: float) -> None:
        self.min_avg_volume = min_avg_volume

    def check(self, inst: Instrument, features: dict[str, float]) -> str | None:
        vol = features.get("vol_avg_20", float("nan"))
        if math.isnan(vol):
            return "avg volume unavailable"
        if vol < self.min_avg_volume:
            return f"avg volume {vol:,.0f} below minimum {self.min_avg_volume:,.0f}"
        return None


class MinHistoryFilter:
    """Rejects instruments with too little price history to trust the features.

    If the caller passes an explicit bar count under the ``n_bars`` feature
    key, that's used directly. Otherwise this falls back to checking that
    ``sma_200`` is finite, which is only true once 200+ bars are available
    (``compute_features`` NaN-guards short histories).
    """

    def __init__(self, min_bars: int) -> None:
        self.min_bars = min_bars

    def check(self, inst: Instrument, features: dict[str, float]) -> str | None:
        n_bars = features.get("n_bars")
        if n_bars is not None:
            if n_bars < self.min_bars:
                return f"only {int(n_bars)} bars of history, need {self.min_bars}"
            return None
        if math.isnan(features.get("sma_200", float("nan"))):
            return f"insufficient history for a {self.min_bars}-bar minimum (sma_200 unavailable)"
        return None


class TrendFilter:
    """Optionally requires ``close`` to be above ``sma_200`` (a broad "is this in
    an uptrend at all" gate, distinct from any strategy's own trend logic)."""

    def __init__(self, require_above_sma200: bool) -> None:
        self.require_above_sma200 = require_above_sma200

    def check(self, inst: Instrument, features: dict[str, float]) -> str | None:
        if not self.require_above_sma200:
            return None
        close = features.get("close", float("nan"))
        sma_200 = features.get("sma_200", float("nan"))
        if math.isnan(close) or math.isnan(sma_200):
            return "insufficient history to evaluate trend vs sma_200"
        if close <= sma_200:
            return f"close {close:.2f} not above sma_200 {sma_200:.2f}"
        return None


class FilterChain:
    """Runs a sequence of ``InstrumentFilter``s, short-circuiting on first rejection."""

    def __init__(self, filters: list[InstrumentFilter]) -> None:
        self._filters = list(filters)

    def run(
        self,
        instruments: list[Instrument],
        features_by_symbol: dict[str, dict[str, float]],
    ) -> tuple[list[Instrument], dict[str, str]]:
        passed: list[Instrument] = []
        rejections: dict[str, str] = {}
        for inst in instruments:
            features = features_by_symbol.get(inst.symbol, {})
            reason = self._evaluate(inst, features)
            if reason is None:
                passed.append(inst)
            else:
                rejections[inst.symbol] = reason
        return passed, rejections

    def _evaluate(self, inst: Instrument, features: dict[str, float]) -> str | None:
        for f in self._filters:
            reason = f.check(inst, features)
            if reason is not None:
                return reason
        return None


def build_default_chain(market: Market) -> FilterChain:
    """Sensible per-market defaults: a liquidity/price floor plus a history minimum.

    US markets (USD): price >= $5, 20-day avg volume >= 500k.
    IN_NSE (INR): price >= Rs 50, 20-day avg volume >= 100k.
    """
    if market.currency == "INR":
        min_price, min_avg_volume = 50.0, 100_000.0
    else:
        min_price, min_avg_volume = 5.0, 500_000.0

    return FilterChain(
        [
            MinPriceFilter(min_price=min_price),
            MinVolumeFilter(min_avg_volume=min_avg_volume),
            MinHistoryFilter(min_bars=200),
        ]
    )
