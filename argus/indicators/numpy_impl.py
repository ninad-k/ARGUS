"""Native numpy indicator implementations.

All functions accept 1-D float64 arrays (oldest-first, length N) and return
``IndicatorResult`` objects with NaN-padded output arrays.

Adapted from public TA formulas (EMA, RSI, MACD etc.) and Jesse's indicator
library (MIT). No verbatim copy - reimplemented from well-known definitions.

Input conventions:
  close  -- closing prices (required by most)
  high   -- session high
  low    -- session low
  volume -- traded volume

All period/length parameters follow the standard convention (length=14 means
the indicator needs 14 bars of history before emitting the first valid value).

Ported from DRUVA's ``core/indicators/numpy_impl.py``; math is kept identical
to the source. Only imports, typing, and this docstring were adapted.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from argus.indicators.base import FloatArray, IndicatorMeta, IndicatorResult

ArrayLike = Sequence[float] | FloatArray


def _arr(x: ArrayLike) -> FloatArray:
    return np.asarray(x, dtype=float)


def _nan_pad(n: int) -> FloatArray:
    return np.full(n, np.nan)


# ---------------------------------------------------------------------------
# Overlap / trend
# ---------------------------------------------------------------------------


def sma(close: ArrayLike, length: int = 20) -> IndicatorResult:
    c = _arr(close)
    n = len(c)
    out = _nan_pad(n)
    for i in range(length - 1, n):
        out[i] = c[i - length + 1 : i + 1].mean()
    return IndicatorResult(
        meta=IndicatorMeta("sma", "Simple Moving Average", "overlap", {"length": length}),
        arrays={"value": out},
    )


def ema(close: ArrayLike, length: int = 20) -> IndicatorResult:
    c = _arr(close)
    n = len(c)
    out = _nan_pad(n)
    if n == 0:
        return IndicatorResult(
            meta=IndicatorMeta(
                "ema", "Exponential Moving Average", "overlap", {"length": length}
            ),
            arrays={"value": out},
        )
    k = 2.0 / (length + 1)
    prev = c[0]
    out[0] = prev
    for i in range(1, n):
        prev = c[i] * k + prev * (1 - k)
        out[i] = prev
    # NaN-pad initial lookback
    out[: length - 1] = np.nan
    return IndicatorResult(
        meta=IndicatorMeta("ema", "Exponential Moving Average", "overlap", {"length": length}),
        arrays={"value": out},
    )


def wma(close: ArrayLike, length: int = 20) -> IndicatorResult:
    """Weighted Moving Average."""
    c = _arr(close)
    n = len(c)
    out = _nan_pad(n)
    weights = np.arange(1, length + 1, dtype=float)
    w_sum = weights.sum()
    for i in range(length - 1, n):
        out[i] = np.dot(c[i - length + 1 : i + 1], weights) / w_sum
    return IndicatorResult(
        meta=IndicatorMeta("wma", "Weighted Moving Average", "overlap", {"length": length}),
        arrays={"value": out},
    )


def dema(close: ArrayLike, length: int = 20) -> IndicatorResult:
    """Double EMA: 2*EMA - EMA(EMA)."""
    e1 = ema(close, length).value
    e2 = ema(e1, length).value
    out = 2 * e1 - e2
    return IndicatorResult(
        meta=IndicatorMeta("dema", "Double EMA", "overlap", {"length": length}),
        arrays={"value": out},
    )


def tema(close: ArrayLike, length: int = 20) -> IndicatorResult:
    """Triple EMA: 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))."""
    e1 = ema(close, length).value
    e2 = ema(e1, length).value
    e3 = ema(e2, length).value
    out = 3 * e1 - 3 * e2 + e3
    return IndicatorResult(
        meta=IndicatorMeta("tema", "Triple EMA", "overlap", {"length": length}),
        arrays={"value": out},
    )


def hma(close: ArrayLike, length: int = 20) -> IndicatorResult:
    """Hull Moving Average: WMA(2*WMA(n/2) - WMA(n), sqrt(n))."""
    c = _arr(close)
    half = max(1, length // 2)
    sqrt_len = max(1, int(np.sqrt(length)))
    w1 = wma(c, half).value
    w2 = wma(c, length).value
    raw = 2 * w1 - w2
    out = wma(raw, sqrt_len).value
    return IndicatorResult(
        meta=IndicatorMeta("hma", "Hull Moving Average", "overlap", {"length": length}),
        arrays={"value": out},
    )


def vwap(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    volume: ArrayLike,
) -> IndicatorResult:
    """Cumulative VWAP (daily reset not implemented - cumulative from bar 0)."""
    h = _arr(high)
    lo = _arr(low)
    c = _arr(close)
    v = _arr(volume)
    tp = (h + lo + c) / 3
    cum_vol = np.cumsum(v)
    cum_tp_vol = np.cumsum(tp * v)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(cum_vol > 0, cum_tp_vol / cum_vol, np.nan)
    return IndicatorResult(
        meta=IndicatorMeta("vwap", "Volume-Weighted Average Price", "overlap"),
        arrays={"value": out},
    )


# ---------------------------------------------------------------------------
# Trend
# ---------------------------------------------------------------------------


def macd(
    close: ArrayLike,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> IndicatorResult:
    fast_ema = ema(close, fast).value
    slow_ema = ema(close, slow).value
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal).value
    histogram = macd_line - signal_line
    return IndicatorResult(
        meta=IndicatorMeta(
            "macd",
            "MACD",
            "trend",
            {"fast": fast, "slow": slow, "signal": signal},
            outputs=("macd", "signal", "histogram"),
        ),
        arrays={"macd": macd_line, "signal": signal_line, "histogram": histogram},
    )


def adx(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    length: int = 14,
) -> IndicatorResult:
    """Average Directional Index (Wilder smoothing)."""
    h = _arr(high)
    lo = _arr(low)
    c = _arr(close)
    n = len(c)
    if n < length + 1:
        nan = _nan_pad(n)
        return IndicatorResult(
            meta=IndicatorMeta(
                "adx",
                "Average Directional Index",
                "trend",
                {"length": length},
                outputs=("adx", "di_plus", "di_minus"),
            ),
            arrays={"adx": nan, "di_plus": nan.copy(), "di_minus": nan.copy()},
        )

    tr = np.maximum(h[1:] - lo[1:], np.maximum(abs(h[1:] - c[:-1]), abs(lo[1:] - c[:-1])))
    dm_plus = np.where(
        (h[1:] - h[:-1]) > (lo[:-1] - lo[1:]), np.maximum(h[1:] - h[:-1], 0), 0.0
    )
    dm_minus = np.where(
        (lo[:-1] - lo[1:]) > (h[1:] - h[:-1]), np.maximum(lo[:-1] - lo[1:], 0), 0.0
    )

    def _wilder(arr: FloatArray) -> FloatArray:
        out = np.zeros(len(arr))
        out[0] = arr[:length].sum()
        for i in range(1, len(arr)):
            out[i] = out[i - 1] - out[i - 1] / length + arr[i]
        return out

    atr_w = _wilder(tr)
    dmp_w = _wilder(dm_plus)
    dmm_w = _wilder(dm_minus)
    with np.errstate(divide="ignore", invalid="ignore"):
        di_plus = np.where(atr_w > 0, dmp_w / atr_w * 100, 0.0)
        di_minus = np.where(atr_w > 0, dmm_w / atr_w * 100, 0.0)
        dx = np.where(
            (di_plus + di_minus) > 0,
            abs(di_plus - di_minus) / (di_plus + di_minus) * 100,
            0.0,
        )

    adx_raw = np.zeros(len(dx))
    adx_raw[length - 1] = dx[:length].mean()
    for i in range(length, len(dx)):
        adx_raw[i] = (adx_raw[i - 1] * (length - 1) + dx[i]) / length

    # Pad back to original length (tr etc. are length n-1)
    pad = np.full(1, np.nan)
    adx_out = np.concatenate([pad, adx_raw])
    dip_out = np.concatenate([pad, di_plus])
    dim_out = np.concatenate([pad, di_minus])
    adx_out[:length] = np.nan
    dip_out[:length] = np.nan
    dim_out[:length] = np.nan

    return IndicatorResult(
        meta=IndicatorMeta(
            "adx",
            "Average Directional Index",
            "trend",
            {"length": length},
            outputs=("adx", "di_plus", "di_minus"),
        ),
        arrays={"adx": adx_out, "di_plus": dip_out, "di_minus": dim_out},
    )


def ichimoku(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    tenkan: int = 9,
    kijun: int = 26,
    senkou_b: int = 52,
    displacement: int = 26,
) -> IndicatorResult:
    """Ichimoku Cloud - returns tenkan, kijun, senkou_a, senkou_b, chikou."""
    h = _arr(high)
    lo = _arr(low)
    c = _arr(close)
    n = len(c)

    def _mid(period: int) -> FloatArray:
        out = _nan_pad(n)
        for i in range(period - 1, n):
            out[i] = (h[i - period + 1 : i + 1].max() + lo[i - period + 1 : i + 1].min()) / 2
        return out

    tenkan_sen = _mid(tenkan)
    kijun_sen = _mid(kijun)
    senkou_a = _nan_pad(n + displacement)
    senkou_b_arr = _nan_pad(n + displacement)
    for i in range(n):
        if not (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i])):
            senkou_a[i + displacement] = (tenkan_sen[i] + kijun_sen[i]) / 2
        if i >= senkou_b - 1:
            senkou_b_arr[i + displacement] = (
                h[i - senkou_b + 1 : i + 1].max() + lo[i - senkou_b + 1 : i + 1].min()
            ) / 2
    chikou = _nan_pad(n)
    for i in range(n - displacement):
        chikou[i] = c[i + displacement]

    return IndicatorResult(
        meta=IndicatorMeta(
            "ichimoku",
            "Ichimoku Cloud",
            "trend",
            {"tenkan": tenkan, "kijun": kijun, "senkou_b": senkou_b},
            outputs=("tenkan", "kijun", "senkou_a", "senkou_b", "chikou"),
        ),
        arrays={
            "tenkan": tenkan_sen,
            "kijun": kijun_sen,
            "senkou_a": senkou_a[:n],
            "senkou_b": senkou_b_arr[:n],
            "chikou": chikou,
        },
    )


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------


def rsi(close: ArrayLike, length: int = 14) -> IndicatorResult:
    c = _arr(close)
    n = len(c)
    out = _nan_pad(n)
    if n < length + 1:
        return IndicatorResult(
            meta=IndicatorMeta("rsi", "Relative Strength Index", "momentum", {"length": length}),
            arrays={"value": out},
        )
    diff = np.diff(c)
    gains = np.where(diff > 0, diff, 0.0)
    losses = np.where(diff < 0, -diff, 0.0)
    avg_gain = gains[:length].mean()
    avg_loss = losses[:length].mean()
    for i in range(length, n - 1):
        avg_gain = (avg_gain * (length - 1) + gains[i]) / length
        avg_loss = (avg_loss * (length - 1) + losses[i]) / length
        if avg_gain == 0 and avg_loss == 0:
            out[i + 1] = np.nan
        elif avg_loss == 0:
            out[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[i + 1] = 100 - 100 / (1 + rs)
    out[:length] = np.nan
    return IndicatorResult(
        meta=IndicatorMeta("rsi", "Relative Strength Index", "momentum", {"length": length}),
        arrays={"value": out},
    )


def stochastic(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    k_length: int = 14,
    d_length: int = 3,
    smooth_k: int = 3,
) -> IndicatorResult:
    """Stochastic Oscillator (%K and %D)."""
    h = _arr(high)
    lo = _arr(low)
    c = _arr(close)
    n = len(c)
    raw_k = _nan_pad(n)
    for i in range(k_length - 1, n):
        h_max = h[i - k_length + 1 : i + 1].max()
        l_min = lo[i - k_length + 1 : i + 1].min()
        if h_max != l_min:
            raw_k[i] = (c[i] - l_min) / (h_max - l_min) * 100
        else:
            raw_k[i] = 50.0
    k = sma(raw_k, smooth_k).value
    d = sma(k, d_length).value
    return IndicatorResult(
        meta=IndicatorMeta(
            "stochastic",
            "Stochastic Oscillator",
            "momentum",
            {"k_length": k_length, "d_length": d_length, "smooth_k": smooth_k},
            outputs=("k", "d"),
        ),
        arrays={"k": k, "d": d},
    )


def cci(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    length: int = 20,
    constant: float = 0.015,
) -> IndicatorResult:
    """Commodity Channel Index."""
    h = _arr(high)
    lo = _arr(low)
    c = _arr(close)
    tp = (h + lo + c) / 3
    n = len(tp)
    out = _nan_pad(n)
    for i in range(length - 1, n):
        window = tp[i - length + 1 : i + 1]
        mean = window.mean()
        mad = np.mean(np.abs(window - mean))
        out[i] = (tp[i] - mean) / (constant * mad) if mad != 0 else 0.0
    return IndicatorResult(
        meta=IndicatorMeta(
            "cci", "Commodity Channel Index", "momentum", {"length": length, "constant": constant}
        ),
        arrays={"value": out},
    )


def roc(close: ArrayLike, length: int = 10) -> IndicatorResult:
    """Rate of Change (%)."""
    c = _arr(close)
    n = len(c)
    out = _nan_pad(n)
    for i in range(length, n):
        if c[i - length] != 0:
            out[i] = (c[i] / c[i - length] - 1) * 100
    return IndicatorResult(
        meta=IndicatorMeta("roc", "Rate of Change", "momentum", {"length": length}),
        arrays={"value": out},
    )


def williams_r(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    length: int = 14,
) -> IndicatorResult:
    """Williams %R."""
    h = _arr(high)
    lo = _arr(low)
    c = _arr(close)
    n = len(c)
    out = _nan_pad(n)
    for i in range(length - 1, n):
        h_max = h[i - length + 1 : i + 1].max()
        l_min = lo[i - length + 1 : i + 1].min()
        if h_max != l_min:
            out[i] = (h_max - c[i]) / (h_max - l_min) * -100
        else:
            out[i] = -50.0
    return IndicatorResult(
        meta=IndicatorMeta("williams_r", "Williams %R", "momentum", {"length": length}),
        arrays={"value": out},
    )


def mfi(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    volume: ArrayLike,
    length: int = 14,
) -> IndicatorResult:
    """Money Flow Index."""
    h = _arr(high)
    lo = _arr(low)
    c = _arr(close)
    v = _arr(volume)
    tp = (h + lo + c) / 3
    mf = tp * v
    n = len(c)
    out = _nan_pad(n)
    for i in range(length, n):
        pos_mf = sum(mf[j] for j in range(i - length + 1, i + 1) if tp[j] > tp[j - 1])
        neg_mf = sum(mf[j] for j in range(i - length + 1, i + 1) if tp[j] < tp[j - 1])
        if neg_mf == 0:
            out[i] = 100.0
        else:
            mfr = pos_mf / neg_mf
            out[i] = 100 - 100 / (1 + mfr)
    return IndicatorResult(
        meta=IndicatorMeta("mfi", "Money Flow Index", "momentum", {"length": length}),
        arrays={"value": out},
    )


def tsi(close: ArrayLike, fast: int = 13, slow: int = 25) -> IndicatorResult:
    """True Strength Index."""
    c = _arr(close)
    diff = np.diff(c)
    ds = np.concatenate([[np.nan], diff])
    abs_ds = np.concatenate([[np.nan], np.abs(diff)])

    def double_smooth(x: FloatArray) -> FloatArray:
        valid = x[~np.isnan(x)]
        e1 = ema(valid, slow).value
        e2 = ema(e1, fast).value
        pad = np.full(len(x) - len(e2), np.nan)
        return np.concatenate([pad, e2])

    num = double_smooth(ds)
    den = double_smooth(abs_ds)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(den != 0, 100 * num / den, np.nan)
    return IndicatorResult(
        meta=IndicatorMeta("tsi", "True Strength Index", "momentum", {"fast": fast, "slow": slow}),
        arrays={"value": out},
    )


def dpo(close: ArrayLike, length: int = 20) -> IndicatorResult:
    """Detrended Price Oscillator."""
    c = _arr(close)
    n = len(c)
    out = _nan_pad(n)
    shift = length // 2 + 1
    s = sma(c, length).value
    for i in range(length - 1, n):
        idx = i - shift
        if idx >= 0:
            out[i] = c[i] - s[idx]
    return IndicatorResult(
        meta=IndicatorMeta("dpo", "Detrended Price Oscillator", "momentum", {"length": length}),
        arrays={"value": out},
    )


def ppo(close: ArrayLike, fast: int = 12, slow: int = 26, signal: int = 9) -> IndicatorResult:
    """Percentage Price Oscillator."""
    fast_e = ema(close, fast).value
    slow_e = ema(close, slow).value
    with np.errstate(divide="ignore", invalid="ignore"):
        ppo_line = np.where(slow_e != 0, (fast_e - slow_e) / slow_e * 100, np.nan)
    sig = ema(ppo_line, signal).value
    hist = ppo_line - sig
    return IndicatorResult(
        meta=IndicatorMeta(
            "ppo",
            "Percentage Price Oscillator",
            "momentum",
            {"fast": fast, "slow": slow, "signal": signal},
            outputs=("ppo", "signal", "histogram"),
        ),
        arrays={"ppo": ppo_line, "signal": sig, "histogram": hist},
    )


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------


def atr(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    length: int = 14,
) -> IndicatorResult:
    h = _arr(high)
    lo = _arr(low)
    c = _arr(close)
    n = len(c)
    tr = np.maximum(h[1:] - lo[1:], np.maximum(abs(h[1:] - c[:-1]), abs(lo[1:] - c[:-1])))
    out = _nan_pad(n)
    out[length] = tr[:length].mean()
    for i in range(length, n - 1):
        out[i + 1] = (out[i] * (length - 1) + tr[i]) / length
    return IndicatorResult(
        meta=IndicatorMeta("atr", "Average True Range", "volatility", {"length": length}),
        arrays={"value": out},
    )


def bollinger_bands(
    close: ArrayLike,
    length: int = 20,
    std_dev: float = 2.0,
) -> IndicatorResult:
    c = _arr(close)
    n = len(c)
    mid = sma(c, length).value
    upper = _nan_pad(n)
    lower = _nan_pad(n)
    for i in range(length - 1, n):
        std = c[i - length + 1 : i + 1].std(ddof=0)
        upper[i] = mid[i] + std_dev * std
        lower[i] = mid[i] - std_dev * std
    return IndicatorResult(
        meta=IndicatorMeta(
            "bbands",
            "Bollinger Bands",
            "volatility",
            {"length": length, "std_dev": std_dev},
            outputs=("upper", "mid", "lower"),
        ),
        arrays={"upper": upper, "mid": mid, "lower": lower},
    )


def keltner_channels(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    ema_length: int = 20,
    atr_length: int = 10,
    multiplier: float = 2.0,
) -> IndicatorResult:
    mid = ema(close, ema_length).value
    atr_val = atr(high, low, close, atr_length).value
    upper = mid + multiplier * atr_val
    lower = mid - multiplier * atr_val
    return IndicatorResult(
        meta=IndicatorMeta(
            "keltner",
            "Keltner Channels",
            "volatility",
            {"ema_length": ema_length, "atr_length": atr_length, "multiplier": multiplier},
            outputs=("upper", "mid", "lower"),
        ),
        arrays={"upper": upper, "mid": mid, "lower": lower},
    )


def donchian_channels(
    high: ArrayLike,
    low: ArrayLike,
    length: int = 20,
) -> IndicatorResult:
    h = _arr(high)
    lo = _arr(low)
    n = len(h)
    upper = _nan_pad(n)
    lower = _nan_pad(n)
    for i in range(length - 1, n):
        upper[i] = h[i - length + 1 : i + 1].max()
        lower[i] = lo[i - length + 1 : i + 1].min()
    mid = (upper + lower) / 2
    return IndicatorResult(
        meta=IndicatorMeta(
            "donchian",
            "Donchian Channels",
            "volatility",
            {"length": length},
            outputs=("upper", "mid", "lower"),
        ),
        arrays={"upper": upper, "mid": mid, "lower": lower},
    )


def historical_volatility(close: ArrayLike, length: int = 20) -> IndicatorResult:
    """Annualised historical volatility (log-return std x sqrt(252))."""
    c = _arr(close)
    n = len(c)
    log_ret = np.log(c[1:] / c[:-1])
    out = _nan_pad(n)
    for i in range(length, n):
        out[i] = log_ret[i - length : i].std(ddof=1) * np.sqrt(252) * 100
    return IndicatorResult(
        meta=IndicatorMeta("hv", "Historical Volatility", "volatility", {"length": length}),
        arrays={"value": out},
    )


def chaikin_volatility(
    high: ArrayLike,
    low: ArrayLike,
    ema_length: int = 10,
    roc_length: int = 10,
) -> IndicatorResult:
    """Chaikin Volatility: ROC of EMA(high-low)."""
    h = _arr(high)
    lo = _arr(low)
    hl = h - lo
    ema_hl = ema(hl, ema_length).value
    n = len(hl)
    out = _nan_pad(n)
    for i in range(roc_length, n):
        if ema_hl[i - roc_length] != 0:
            out[i] = (ema_hl[i] - ema_hl[i - roc_length]) / ema_hl[i - roc_length] * 100
    return IndicatorResult(
        meta=IndicatorMeta(
            "chaikin_vol",
            "Chaikin Volatility",
            "volatility",
            {"ema_length": ema_length, "roc_length": roc_length},
        ),
        arrays={"value": out},
    )


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------


def obv(close: ArrayLike, volume: ArrayLike) -> IndicatorResult:
    c = _arr(close)
    v = _arr(volume)
    n = len(c)
    out = _nan_pad(n)
    out[0] = v[0]
    for i in range(1, n):
        out[i] = out[i - 1] + (v[i] if c[i] > c[i - 1] else (-v[i] if c[i] < c[i - 1] else 0))
    return IndicatorResult(
        meta=IndicatorMeta("obv", "On-Balance Volume", "volume"),
        arrays={"value": out},
    )


def vwma(close: ArrayLike, volume: ArrayLike, length: int = 20) -> IndicatorResult:
    """Volume-Weighted Moving Average."""
    c = _arr(close)
    v = _arr(volume)
    n = len(c)
    out = _nan_pad(n)
    for i in range(length - 1, n):
        cv = c[i - length + 1 : i + 1]
        vv = v[i - length + 1 : i + 1]
        out[i] = np.dot(cv, vv) / vv.sum() if vv.sum() != 0 else np.nan
    return IndicatorResult(
        meta=IndicatorMeta("vwma", "Volume-Weighted MA", "volume", {"length": length}),
        arrays={"value": out},
    )


def chaikin_mf(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    volume: ArrayLike,
    length: int = 20,
) -> IndicatorResult:
    """Chaikin Money Flow."""
    h = _arr(high)
    lo = _arr(low)
    c = _arr(close)
    v = _arr(volume)
    with np.errstate(divide="ignore", invalid="ignore"):
        mfm = np.where((h - lo) != 0, (c - lo - (h - c)) / (h - lo), 0.0)
    mfv = mfm * v
    n = len(c)
    out = _nan_pad(n)
    for i in range(length - 1, n):
        vol_sum = v[i - length + 1 : i + 1].sum()
        out[i] = mfv[i - length + 1 : i + 1].sum() / vol_sum if vol_sum != 0 else np.nan
    return IndicatorResult(
        meta=IndicatorMeta("cmf", "Chaikin Money Flow", "volume", {"length": length}),
        arrays={"value": out},
    )


def ease_of_movement(
    high: ArrayLike,
    low: ArrayLike,
    volume: ArrayLike,
    length: int = 14,
) -> IndicatorResult:
    """Ease of Movement."""
    h = _arr(high)
    lo = _arr(low)
    v = _arr(volume)
    n = len(h)
    raw = _nan_pad(n)
    for i in range(1, n):
        hl = h[i] - lo[i]
        if hl != 0 and v[i] != 0:
            dm = (h[i] + lo[i]) / 2 - (h[i - 1] + lo[i - 1]) / 2
            box = v[i] / 1e6 / hl
            raw[i] = dm / box if box != 0 else np.nan
    out = sma(raw, length).value
    return IndicatorResult(
        meta=IndicatorMeta("emv", "Ease of Movement", "volume", {"length": length}),
        arrays={"value": out},
    )


def force_index(
    close: ArrayLike,
    volume: ArrayLike,
    length: int = 13,
) -> IndicatorResult:
    """Elder Force Index."""
    c = _arr(close)
    v = _arr(volume)
    n = len(c)
    raw = _nan_pad(n)
    for i in range(1, n):
        raw[i] = (c[i] - c[i - 1]) * v[i]
    out = ema(raw, length).value
    return IndicatorResult(
        meta=IndicatorMeta("force_index", "Force Index", "volume", {"length": length}),
        arrays={"value": out},
    )


# ---------------------------------------------------------------------------
# Cycle / other
# ---------------------------------------------------------------------------


def aroon(
    high: ArrayLike,
    low: ArrayLike,
    length: int = 25,
) -> IndicatorResult:
    """Aroon Oscillator."""
    h = _arr(high)
    lo = _arr(low)
    n = len(h)
    up = _nan_pad(n)
    down = _nan_pad(n)
    for i in range(length, n):
        window_h = h[i - length : i + 1]
        window_l = lo[i - length : i + 1]
        up[i] = (np.argmax(window_h[::-1]) == 0) * 100 + (
            length - np.argmax(window_h[::-1])
        ) / length * 100
        # Recalculate cleanly
        up[i] = (length - (length - np.argmax(window_h))) / length * 100
        down[i] = (length - (length - np.argmin(window_l))) / length * 100
    oscillator = up - down
    return IndicatorResult(
        meta=IndicatorMeta(
            "aroon", "Aroon", "cycle", {"length": length}, outputs=("up", "down", "oscillator")
        ),
        arrays={"up": up, "down": down, "oscillator": oscillator},
    )


def supertrend(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    atr_length: int = 7,
    multiplier: float = 3.0,
) -> IndicatorResult:
    """Supertrend indicator."""
    h = _arr(high)
    lo = _arr(low)
    c = _arr(close)
    n = len(c)
    atr_val = atr(h, lo, c, atr_length).value
    hl2 = (h + lo) / 2
    upper_basic = hl2 + multiplier * atr_val
    lower_basic = hl2 - multiplier * atr_val

    upper = _nan_pad(n)
    lower = _nan_pad(n)
    st = _nan_pad(n)
    direction = _nan_pad(n)

    for i in range(atr_length, n):
        upper[i] = (
            upper_basic[i] if (upper_basic[i] < upper[i - 1] or c[i - 1] > upper[i - 1])
            else upper[i - 1]
        )
        lower[i] = (
            lower_basic[i] if (lower_basic[i] > lower[i - 1] or c[i - 1] < lower[i - 1])
            else lower[i - 1]
        )
        if np.isnan(st[i - 1]):
            st[i] = upper[i]
            direction[i] = -1
        elif st[i - 1] == upper[i - 1]:
            st[i] = lower[i] if c[i] > upper[i] else upper[i]
            direction[i] = 1 if c[i] > upper[i] else -1
        else:
            st[i] = upper[i] if c[i] < lower[i] else lower[i]
            direction[i] = -1 if c[i] < lower[i] else 1

    return IndicatorResult(
        meta=IndicatorMeta(
            "supertrend",
            "Supertrend",
            "trend",
            {"atr_length": atr_length, "multiplier": multiplier},
            outputs=("supertrend", "direction"),
        ),
        arrays={"supertrend": st, "direction": direction},
    )


def pivot_points(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> IndicatorResult:
    """Classic pivot points (previous-bar calculation)."""
    h = _arr(high)
    lo = _arr(low)
    c = _arr(close)
    n = len(c)
    pp = _nan_pad(n)
    r1 = _nan_pad(n)
    s1 = _nan_pad(n)
    r2 = _nan_pad(n)
    s2 = _nan_pad(n)
    for i in range(1, n):
        pp[i] = (h[i - 1] + lo[i - 1] + c[i - 1]) / 3
        r1[i] = 2 * pp[i] - lo[i - 1]
        s1[i] = 2 * pp[i] - h[i - 1]
        r2[i] = pp[i] + (h[i - 1] - lo[i - 1])
        s2[i] = pp[i] - (h[i - 1] - lo[i - 1])
    return IndicatorResult(
        meta=IndicatorMeta(
            "pivot", "Pivot Points", "other", outputs=("pp", "r1", "r2", "s1", "s2")
        ),
        arrays={"pp": pp, "r1": r1, "r2": r2, "s1": s1, "s2": s2},
    )
