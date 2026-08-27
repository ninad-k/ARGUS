"""Fundamentals data: protocol, providers, and the kind->provider factory."""

from argus.data.fundamentals.base import (
    FundamentalsProvider,
    FundamentalsView,
    NullFundamentalsProvider,
    StaticFundamentalsProvider,
    default_get_many,
)
from argus.data.fundamentals.factory import build_default_fundamentals, build_fundamentals_provider
from argus.data.fundamentals.tv_fundamentals import TVScreenerFundamentalsProvider
from argus.data.fundamentals.yfinance_fundamentals import YFinanceFundamentalsProvider

__all__ = [
    "FundamentalsProvider",
    "FundamentalsView",
    "NullFundamentalsProvider",
    "StaticFundamentalsProvider",
    "TVScreenerFundamentalsProvider",
    "YFinanceFundamentalsProvider",
    "build_default_fundamentals",
    "build_fundamentals_provider",
    "default_get_many",
]
