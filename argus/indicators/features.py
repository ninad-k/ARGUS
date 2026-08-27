"""Convenience feature layer for the screener.

``compute_features`` takes a ``BAR_DTYPE`` structured array (ascending by
timestamp, oldest-first) and returns the latest-bar scalar features the
screener strategies consume. Values are ``float("nan")`` when the input
history is too short for a given feature, rather than raising.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from argus.indicators import numpy_impl as _ni

FEATURE_KEYS: tuple[str, ...] = (
    "close",
    "sma_20",
    "sma_50",
    "sma_200",
    "ema_20",
    "rsi_14",
    "atr_14",
    "adx_14",
    "roc_20",
    "roc_60",
    "bb_width_20",
    "vol_avg_20",
    "rvol",
    "high_252",
    "pct_off_high",
    "dist_sma200_pct",
)

_NAN_FEATURES: dict[str, float] = dict.fromkeys(FEATURE_KEYS, float("nan"))


def _last(arr: NDArray[np.float64]) -> float:
    """Latest value of an indicator output array, as a plain float."""
    return float(arr[-1])


def _ratio(a: float, b: float) -> float:
    """``a / b``, guarding against division by zero or NaN (which raise on
    plain Python floats rather than propagating like numpy does)."""
    if b == 0 or math.isnan(b):
        return float("nan")
    return a / b


def compute_features(bars: NDArray[np.void]) -> dict[str, float]:
    """Compute latest-bar scalar features from a ``BAR_DTYPE`` bars array.

    Guards for short histories: any feature that needs more bars than are
    available comes back as NaN instead of raising.
    """
    n = len(bars)
    if n == 0:
        return dict(_NAN_FEATURES)

    close = bars["close"].astype(np.float64)
    high = bars["high"].astype(np.float64)
    low = bars["low"].astype(np.float64)
    volume = bars["volume"].astype(np.float64)

    last_close = float(close[-1])
    last_volume = float(volume[-1])

    sma_20 = _last(_ni.sma(close, 20).value)
    sma_50 = _last(_ni.sma(close, 50).value)
    sma_200 = _last(_ni.sma(close, 200).value)
    ema_20 = _last(_ni.ema(close, 20).value)
    rsi_14 = _last(_ni.rsi(close, 14).value)
    # atr() indexes out[length] unconditionally, which is out of bounds when
    # n <= length -- gate the call rather than reimplementing the math.
    atr_14 = _last(_ni.atr(high, low, close, 14).value) if n > 14 else float("nan")
    adx_14 = _last(_ni.adx(high, low, close, 14).value)
    roc_20 = _last(_ni.roc(close, 20).value)
    roc_60 = _last(_ni.roc(close, 60).value)

    bb = _ni.bollinger_bands(close, 20, 2.0)
    bb_upper = _last(bb.arrays["upper"])
    bb_lower = _last(bb.arrays["lower"])
    bb_mid = _last(bb.arrays["mid"])
    bb_width_20 = _ratio(bb_upper - bb_lower, bb_mid)

    vol_avg_20 = _last(_ni.sma(volume, 20).value)
    rvol = _ratio(last_volume, vol_avg_20)

    high_252 = _last(_ni.donchian_channels(high, low, 252).arrays["upper"])
    pct_off_high = _ratio(last_close - high_252, high_252) * 100

    dist_sma200_pct = _ratio(last_close - sma_200, sma_200) * 100

    return {
        "close": last_close,
        "sma_20": sma_20,
        "sma_50": sma_50,
        "sma_200": sma_200,
        "ema_20": ema_20,
        "rsi_14": rsi_14,
        "atr_14": atr_14,
        "adx_14": adx_14,
        "roc_20": roc_20,
        "roc_60": roc_60,
        "bb_width_20": bb_width_20,
        "vol_avg_20": vol_avg_20,
        "rvol": rvol,
        "high_252": high_252,
        "pct_off_high": pct_off_high,
        "dist_sma200_pct": dist_sma200_pct,
    }
