"""Per-filter pass/reject behavior, chain short-circuiting, and market defaults."""

import math

from argus.markets import IN_NSE, US_NASDAQ, Instrument
from argus.screener.filters import (
    FilterChain,
    MinHistoryFilter,
    MinPriceFilter,
    MinVolumeFilter,
    TrendFilter,
    build_default_chain,
)

_INST = Instrument(symbol="AAPL", market_code=US_NASDAQ.code)


def test_min_price_filter_passes_above_threshold() -> None:
    f = MinPriceFilter(min_price=10.0)
    assert f.check(_INST, {"close": 10.0}) is None
    assert f.check(_INST, {"close": 50.0}) is None


def test_min_price_filter_rejects_below_threshold() -> None:
    f = MinPriceFilter(min_price=10.0)
    reason = f.check(_INST, {"close": 9.99})
    assert reason is not None
    assert "9.99" in reason


def test_min_price_filter_rejects_missing_price() -> None:
    f = MinPriceFilter(min_price=10.0)
    assert f.check(_INST, {"close": float("nan")}) is not None
    assert f.check(_INST, {}) is not None


def test_min_volume_filter_passes_and_rejects() -> None:
    f = MinVolumeFilter(min_avg_volume=500_000)
    assert f.check(_INST, {"vol_avg_20": 500_000}) is None
    reason = f.check(_INST, {"vol_avg_20": 499_999})
    assert reason is not None
    assert "499,999" in reason


def test_min_history_filter_uses_n_bars_when_present() -> None:
    f = MinHistoryFilter(min_bars=200)
    assert f.check(_INST, {"n_bars": 200}) is None
    reason = f.check(_INST, {"n_bars": 199})
    assert reason is not None
    assert "199" in reason


def test_min_history_filter_falls_back_to_sma_200_finiteness() -> None:
    f = MinHistoryFilter(min_bars=200)
    assert f.check(_INST, {"sma_200": 123.4}) is None
    assert f.check(_INST, {"sma_200": float("nan")}) is not None
    assert f.check(_INST, {}) is not None


def test_trend_filter_disabled_always_passes() -> None:
    f = TrendFilter(require_above_sma200=False)
    assert f.check(_INST, {"close": 1.0, "sma_200": 100.0}) is None


def test_trend_filter_enabled_requires_close_above_sma200() -> None:
    f = TrendFilter(require_above_sma200=True)
    assert f.check(_INST, {"close": 110.0, "sma_200": 100.0}) is None
    reason = f.check(_INST, {"close": 90.0, "sma_200": 100.0})
    assert reason is not None


def test_trend_filter_enabled_rejects_nan_inputs() -> None:
    f = TrendFilter(require_above_sma200=True)
    assert f.check(_INST, {"close": float("nan"), "sma_200": 100.0}) is not None


def test_filter_chain_short_circuits_on_first_rejection() -> None:
    chain = FilterChain(
        [MinPriceFilter(min_price=10.0), MinVolumeFilter(min_avg_volume=1_000)]
    )
    features = {"close": 1.0, "vol_avg_20": 1.0}  # fails both
    passed, rejections = chain.run([_INST], {"AAPL": features})

    assert passed == []
    assert "AAPL" in rejections
    # Only the first (price) filter's reason should appear -- volume was
    # never evaluated because the chain short-circuited.
    assert "price" in rejections["AAPL"]
    assert "volume" not in rejections["AAPL"]


def test_filter_chain_passes_instrument_clearing_all_filters() -> None:
    chain = FilterChain(
        [MinPriceFilter(min_price=10.0), MinVolumeFilter(min_avg_volume=1_000)]
    )
    features = {"close": 100.0, "vol_avg_20": 10_000.0}
    passed, rejections = chain.run([_INST], {"AAPL": features})

    assert passed == [_INST]
    assert rejections == {}


def test_filter_chain_reports_missing_features_as_rejection() -> None:
    chain = FilterChain([MinPriceFilter(min_price=10.0)])
    passed, rejections = chain.run([_INST], {})  # no features recorded at all

    assert passed == []
    assert "AAPL" in rejections


def test_build_default_chain_us_thresholds() -> None:
    chain = build_default_chain(US_NASDAQ)

    ok_features = {"close": 5.0, "vol_avg_20": 500_000.0, "sma_200": 1.0}
    passed, _ = chain.run([_INST], {"AAPL": ok_features})
    assert passed == [_INST]

    bad_price = {"close": 4.99, "vol_avg_20": 500_000.0, "sma_200": 1.0}
    passed, rejections = chain.run([_INST], {"AAPL": bad_price})
    assert passed == []
    assert math.isfinite(4.99)  # sanity: not testing a NaN edge case here


def test_build_default_chain_in_nse_thresholds() -> None:
    chain = build_default_chain(IN_NSE)
    inst = Instrument(symbol="RELIANCE", market_code=IN_NSE.code)

    ok_features = {"close": 50.0, "vol_avg_20": 100_000.0, "sma_200": 1.0}
    passed, _ = chain.run([inst], {"RELIANCE": ok_features})
    assert passed == [inst]

    bad_volume = {"close": 50.0, "vol_avg_20": 99_999.0, "sma_200": 1.0}
    passed, rejections = chain.run([inst], {"RELIANCE": bad_volume})
    assert passed == []
    assert "RELIANCE" in rejections
