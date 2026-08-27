"""Base types for the indicator library.

Every indicator is a callable that accepts a numpy array (or list) of OHLCV
data and returns a numpy array of the same length, right-aligned (the first
``lookback`` values are NaN where the indicator has insufficient history).

``IndicatorMeta`` carries the metadata used by the registry. ``IndicatorResult``
wraps the raw array with column labels for multi-output indicators (MACD
returns 3 series, Bollinger returns 3 bands, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class IndicatorMeta:
    """Registry record for one indicator."""

    name: str
    """Lower-snake-case canonical name, e.g. ``"ema"``."""

    display_name: str
    """Human-readable label, e.g. ``"Exponential Moving Average"``."""

    category: str
    """One of: trend, momentum, volatility, volume, overlap, cycle, other."""

    params: dict[str, Any] = field(default_factory=dict)
    """Default parameter values (used when caller omits them)."""

    outputs: tuple[str, ...] = ("value",)
    """Names of output series for multi-output indicators."""

    description: str = ""


@dataclass
class IndicatorResult:
    """Wraps raw numpy arrays with column labels."""

    meta: IndicatorMeta
    arrays: dict[str, FloatArray]
    """Maps output name -> 1-D float64 array of length N (NaN-padded)."""

    @property
    def value(self) -> FloatArray:
        """Convenience accessor for single-output indicators."""
        return self.arrays[self.meta.outputs[0]]

    def to_dict(self) -> dict[str, list[float | None]]:
        """Serialise to a JSON-friendly dict, replacing NaN with None."""
        return {
            k: [None if np.isnan(v) else float(v) for v in arr] for k, arr in self.arrays.items()
        }
