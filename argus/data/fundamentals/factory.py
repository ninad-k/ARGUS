"""Factory for building a ``FundamentalsProvider`` from a kind/config pair.

Mirrors ``argus.data.sources.build_price_provider``'s shape so the sources
admin UI/API can eventually configure a fundamentals source the same way it
configures a price source.
"""

from __future__ import annotations

from typing import Any

import structlog

from argus.data.fundamentals.base import (
    FundamentalsProvider,
    NullFundamentalsProvider,
    StaticFundamentalsProvider,
)
from argus.data.fundamentals.tv_fundamentals import TVScreenerFundamentalsProvider
from argus.data.fundamentals.yfinance_fundamentals import YFinanceFundamentalsProvider

logger = structlog.get_logger(__name__)


def build_fundamentals_provider(kind: str, config: dict[str, Any]) -> FundamentalsProvider:
    """Construct a ``FundamentalsProvider`` from a kind + config dict.

    Falls back to ``NullFundamentalsProvider`` for unknown kinds -- never raises.
    """
    if kind == "yfinance":
        return YFinanceFundamentalsProvider()
    if kind == "tvscreener":
        return TVScreenerFundamentalsProvider()
    if kind == "static":
        return StaticFundamentalsProvider()
    if kind == "null":
        return NullFundamentalsProvider()
    logger.warning("fundamentals.factory.unknown_kind", kind=kind)
    return NullFundamentalsProvider()


def build_default_fundamentals() -> FundamentalsProvider:
    """The pipeline's default fundamentals provider (Yahoo Finance)."""
    return YFinanceFundamentalsProvider()
