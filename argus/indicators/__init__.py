"""Pure-numpy technical indicator library and screener feature layer.

Ported from DRUVA's ``core/indicators`` package.
"""

from argus.indicators.base import FloatArray, IndicatorMeta, IndicatorResult
from argus.indicators.features import FEATURE_KEYS, compute_features
from argus.indicators.registry import IndicatorRegistry, get_indicator, list_indicators, registry

__all__ = [
    "FEATURE_KEYS",
    "FloatArray",
    "IndicatorMeta",
    "IndicatorRegistry",
    "IndicatorResult",
    "compute_features",
    "get_indicator",
    "list_indicators",
    "registry",
]
