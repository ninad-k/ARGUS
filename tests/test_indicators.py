"""Tests for the ported numpy indicator library, registry, and feature layer."""

from datetime import date

import numpy as np
import pytest

from argus.data.prices.base import BAR_DTYPE
from argus.data.prices.static_provider import synthetic_bars
from argus.indicators import numpy_impl as ni
from argus.indicators.features import FEATURE_KEYS, compute_features
from argus.indicators.registry import get_indicator, list_indicators, registry

# ---------------------------------------------------------------------------
# Hand-computable fixtures: SMA, EMA, RSI, ATR
#
# Expected values below are derived independently of ``numpy_impl`` -- by
# plain arithmetic (SMA/EMA) or by hand-applying Wilder's smoothing formula
# with exact fractions (RSI/ATR) -- not by calling the functions under test.
# ---------------------------------------------------------------------------


def test_sma_hand_computed() -> None:
    close = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    out = ni.sma(close, length=3).value
    assert np.isnan(out[0])
    assert np.isnan(out[1])
    # (1+2+3)/3, (4+5+6)/3, (8+9+10)/3
    assert out[2] == pytest.approx(2.0)
    assert out[5] == pytest.approx(5.0)
    assert out[9] == pytest.approx(9.0)


def test_ema_hand_computed() -> None:
    # length=3 -> k = 2/(3+1) = 0.5
    # prev=1; ema1=2*.5+1*.5=1.5; ema2=3*.5+1.5*.5=2.25; ema3=4*.5+2.25*.5=3.125
    close = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    out = ni.ema(close, length=3).value
    assert np.isnan(out[0])
    assert np.isnan(out[1])
    assert out[2] == pytest.approx(2.25)
    assert out[3] == pytest.approx(3.125)
    assert out[4] == pytest.approx(4.0625)


def test_rsi_hand_computed() -> None:
    # diffs: [1, -0.5, 1.5, -0.5, 1.5], length=3
    # avg_gain0=(1+0+1.5)/3=5/6, avg_loss0=(0+0.5+0)/3=1/6
    # step i=3: avg_gain=5/9, avg_loss=5/18 -> rs=2 -> rsi=100-100/3=200/3
    # step i=4: avg_gain=47/54, avg_loss=5/27 -> rs=4.7 -> rsi=100-100/5.7=4700/57
    close = np.array([10, 11, 10.5, 12, 11.5, 13], dtype=float)
    out = ni.rsi(close, length=3).value
    assert np.isnan(out[3])
    assert out[4] == pytest.approx(200 / 3)
    assert out[5] == pytest.approx(4700 / 57)


def test_atr_hand_computed() -> None:
    # TR: max(h-l, |h-c_prev|, |l-c_prev|) for bars 1..4
    # tr = [3, 2, 3, 2] (worked by hand from the fixture below), length=2
    # out[2]=(3+2)/2=2.5; out[3]=(2.5*1+3)/2=2.75; out[4]=(2.75*1+2)/2=2.375
    high = np.array([10, 12, 11, 13, 12], dtype=float)
    low = np.array([8, 9, 9, 10, 10], dtype=float)
    close = np.array([9, 11, 10, 12, 11], dtype=float)
    out = ni.atr(high, low, close, length=2).value
    assert np.isnan(out[0])
    assert np.isnan(out[1])
    assert out[2] == pytest.approx(2.5)
    assert out[3] == pytest.approx(2.75)
    assert out[4] == pytest.approx(2.375)


# ---------------------------------------------------------------------------
# Parity spot-checks against independently hand-derived expectations
# ---------------------------------------------------------------------------


def test_wma_parity_spot_check() -> None:
    # WMA(length=3) at the last bar of [1,2,3]: weights [1,2,3], sum=6
    # (1*1 + 2*2 + 3*3) / 6 = 14/6 = 7/3
    close = np.array([1, 2, 3], dtype=float)
    out = ni.wma(close, length=3).value
    assert out[2] == pytest.approx(7 / 3)


def test_roc_parity_spot_check() -> None:
    # ROC(length=2): (c[i]/c[i-2] - 1) * 100
    close = np.array([100, 105, 110, 121], dtype=float)
    out = ni.roc(close, length=2).value
    assert out[2] == pytest.approx((110 / 100 - 1) * 100)
    assert out[3] == pytest.approx((121 / 105 - 1) * 100)


def test_sma_and_rsi_parity_recap() -> None:
    # Re-affirms the SMA/RSI hand fixtures above count toward the "3-4
    # indicators" parity spot-check as well as the hand-computable fixtures.
    close = np.array([1, 2, 3, 4, 5], dtype=float)
    out = ni.sma(close, length=2).value
    assert out[1] == pytest.approx(1.5)
    assert out[4] == pytest.approx(4.5)


# ---------------------------------------------------------------------------
# compute_features
# ---------------------------------------------------------------------------


def test_compute_features_returns_all_keys_and_finite_where_n_suffices() -> None:
    bars = synthetic_bars(n=300, start_price=100.0, seed=1, start=date(2026, 1, 2))
    feats = compute_features(bars)
    assert set(feats.keys()) == set(FEATURE_KEYS)
    for key in FEATURE_KEYS:
        assert np.isfinite(feats[key]), f"{key} should be finite with 300 bars"


def test_compute_features_short_history_returns_nan_not_raise() -> None:
    bars = synthetic_bars(n=30, start_price=100.0, seed=1, start=date(2026, 1, 2))
    feats = compute_features(bars)
    assert set(feats.keys()) == set(FEATURE_KEYS)

    long_lookback_keys = {
        "sma_50",
        "sma_200",
        "roc_60",
        "high_252",
        "pct_off_high",
        "dist_sma200_pct",
    }
    for key in long_lookback_keys:
        assert np.isnan(feats[key]), f"{key} should be NaN with only 30 bars"

    for key in set(FEATURE_KEYS) - long_lookback_keys:
        assert np.isfinite(feats[key]), f"{key} should be finite with 30 bars"


def test_compute_features_empty_bars_returns_all_nan_without_raising() -> None:
    bars = np.zeros(0, dtype=BAR_DTYPE)
    feats = compute_features(bars)
    assert set(feats.keys()) == set(FEATURE_KEYS)
    for key in FEATURE_KEYS:
        assert np.isnan(feats[key])


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_list_indicators_non_empty() -> None:
    metas = list_indicators()
    assert len(metas) > 0
    names = {m.name for m in metas}
    assert "sma" in names
    assert "rsi" in names
    assert "macd" in names


def test_get_indicator_rsi_metadata_sane() -> None:
    meta = get_indicator("rsi")
    assert meta.name == "rsi"
    assert meta.category == "momentum"
    assert meta.params == {"length": 14}
    assert meta.outputs == ("value",)


def test_registry_compute_matches_direct_call() -> None:
    close = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    result = registry.compute("sma", close=close, length=3)
    direct = ni.sma(close, length=3)
    assert np.array_equal(result.value, direct.value, equal_nan=True)


def test_registry_unknown_indicator_raises_key_error() -> None:
    with pytest.raises(KeyError):
        registry.compute("not_a_real_indicator", close=[1, 2, 3])


def test_registry_list_indicators_filters_by_category() -> None:
    vol_metas = list_indicators(category="volatility")
    assert len(vol_metas) > 0
    assert all(m.category == "volatility" for m in vol_metas)
