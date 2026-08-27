"""Full ``compute_orderflow`` assembly, with and without an option chain, and
``to_feature_dict`` JSON-serializability."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta

import numpy as np
from numpy.typing import NDArray

from argus.data.prices.base import bars_from_columns
from argus.options.models import OptionChain, OptionQuote
from argus.orderflow.features import compute_orderflow, to_feature_dict

_START = date(2026, 1, 2)
_EXPIRY = date(2026, 3, 20)


def _daily_bars(n: int = 40, base: float = 100.0) -> NDArray[np.void]:
    """A mild uptrend with enough history for compute_features' rvol/atr, an
    unfilled gap-up on the last bar, and a reclaimed low sweep two bars back."""
    rng = np.random.RandomState(7)
    closes = base + np.cumsum(rng.normal(loc=0.2, scale=0.5, size=n))
    opens = np.empty(n)
    opens[0] = base
    opens[1:] = closes[:-1]
    highs = np.maximum(opens, closes) + 0.5
    lows = np.minimum(opens, closes) - 0.5
    volumes = np.full(n, 1_000_000.0)

    # Reclaimed low sweep two sessions back: trades below the trailing 20d
    # low but closes back inside.
    lows[-3] = min(lows[:-3]) - 5.0
    closes[-3] = closes[-4]
    opens[-3] = closes[-4]
    highs[-3] = max(opens[-3], closes[-3]) + 0.5

    # Unfilled gap-up on the last bar.
    prev_close = closes[-2]
    opens[-1] = prev_close * 1.02
    closes[-1] = opens[-1] * 1.01
    lows[-1] = opens[-1] * 0.999  # stays well above prev_close -- unfilled
    highs[-1] = closes[-1] * 1.005
    volumes[-1] = volumes[-2] * 3  # elevated rvol

    ts = np.array(
        [np.datetime64((_START + timedelta(days=i)).isoformat(), "s") for i in range(n)],
        dtype="datetime64[s]",
    )
    return bars_from_columns(ts, opens, highs, lows, closes, volumes)


def _chain(spot: float) -> OptionChain:
    quotes = [
        OptionQuote(strike=95.0, expiry=_EXPIRY, right="C", oi=500.0, gamma=0.05),
        OptionQuote(strike=95.0, expiry=_EXPIRY, right="P", oi=200.0, gamma=0.05),
        OptionQuote(strike=100.0, expiry=_EXPIRY, right="C", oi=800.0, gamma=0.08),
        OptionQuote(strike=100.0, expiry=_EXPIRY, right="P", oi=600.0, gamma=0.08),
        OptionQuote(strike=105.0, expiry=_EXPIRY, right="C", oi=300.0, gamma=0.04),
        OptionQuote(strike=105.0, expiry=_EXPIRY, right="P", oi=900.0, gamma=0.04),
    ]
    return OptionChain(
        symbol="TEST",
        market_code="US_NASDAQ",
        spot=spot,
        as_of=datetime.now(UTC),
        expiries=[_EXPIRY],
        quotes=quotes,
    )


def test_compute_orderflow_without_chain_has_none_options_fields() -> None:
    bars = _daily_bars()

    of = compute_orderflow(bars)

    assert of is not None
    assert of.pcr_oi is None
    assert of.gex_sign is None
    assert of.max_pain_dist_pct is None
    # OHLCV-derived fields should still be populated.
    assert of.vp_poc is not None
    assert of.rvol is not None
    assert of.gap_kind == "gap_up"
    assert of.gap_filled is False


def test_compute_orderflow_with_chain_populates_options_fields() -> None:
    bars = _daily_bars()
    chain = _chain(spot=float(bars[-1]["close"]))

    of = compute_orderflow(bars, chain=chain)

    assert of is not None
    assert of.pcr_oi is not None
    assert of.gex_sign in {"long_gamma", "short_gamma"}
    assert of.max_pain_dist_pct is not None


def test_compute_orderflow_uses_intraday_bars_for_volume_profile_when_given() -> None:
    daily = _daily_bars()
    # A narrow intraday range should produce a POC well inside the daily
    # bars' much wider range.
    intraday_n = 10
    intraday_closes = np.linspace(99.0, 101.0, intraday_n)
    intraday_ts = np.array(
        [np.datetime64((_START + timedelta(hours=i)).isoformat(), "s") for i in range(intraday_n)],
        dtype="datetime64[s]",
    )
    intraday = bars_from_columns(
        intraday_ts,
        intraday_closes,
        intraday_closes + 0.1,
        intraday_closes - 0.1,
        intraday_closes,
        np.full(intraday_n, 10_000.0),
    )

    of_daily_only = compute_orderflow(daily)
    of_with_intraday = compute_orderflow(daily, intraday_bars=intraday)

    assert of_daily_only is not None
    assert of_with_intraday is not None
    vp_poc = of_with_intraday.vp_poc
    assert vp_poc is not None
    assert vp_poc != of_daily_only.vp_poc
    assert 98.5 <= vp_poc <= 101.5


def test_compute_orderflow_empty_bars_returns_none() -> None:
    empty = bars_from_columns(
        np.array([], dtype="datetime64[s]"),
        np.array([]),
        np.array([]),
        np.array([]),
        np.array([]),
        np.array([]),
    )
    assert compute_orderflow(empty) is None


def test_to_feature_dict_is_json_serializable() -> None:
    bars = _daily_bars()
    chain = _chain(spot=float(bars[-1]["close"]))
    of = compute_orderflow(bars, chain=chain)
    assert of is not None

    payload = to_feature_dict(of)
    encoded = json.dumps(payload)
    decoded = json.loads(encoded)

    assert decoded["gap_kind"] == of.gap_kind
    assert decoded["pcr_oi"] == of.pcr_oi
