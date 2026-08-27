"""Unified screener engine: Strategy ABC, registry, filter chain, batch runner."""

from argus.screener.base import Candidate, DefaultScreenContext, ScreenContext, Strategy
from argus.screener.filters import (
    FilterChain,
    InstrumentFilter,
    MinHistoryFilter,
    MinPriceFilter,
    MinVolumeFilter,
    TrendFilter,
    build_default_chain,
)
from argus.screener.registry import all_strategies, get_strategy, register_strategy
from argus.screener.runner import ScreenResult, persist_screen_result, run_screen

__all__ = [
    "Candidate",
    "DefaultScreenContext",
    "FilterChain",
    "InstrumentFilter",
    "MinHistoryFilter",
    "MinPriceFilter",
    "MinVolumeFilter",
    "ScreenContext",
    "ScreenResult",
    "Strategy",
    "TrendFilter",
    "all_strategies",
    "build_default_chain",
    "get_strategy",
    "persist_screen_result",
    "register_strategy",
    "run_screen",
]
