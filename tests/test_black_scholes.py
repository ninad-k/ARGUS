"""Black-Scholes price/greeks vs. known reference values (S=100, K=100,
T=1, r=5%, sigma=20% -- a standard textbook example), plus an
implied_vol price -> iv -> price round-trip."""

from __future__ import annotations

import math

import pytest

from argus.options import black_scholes

_S = 100.0
_K = 100.0
_T = 1.0
_R = 0.05
_SIGMA = 0.20


def test_call_price_matches_reference() -> None:
    p = black_scholes.price(_S, _K, _T, _SIGMA, "C", _R)
    assert p == pytest.approx(10.4506, abs=1e-3)


def test_put_price_matches_reference() -> None:
    p = black_scholes.price(_S, _K, _T, _SIGMA, "P", _R)
    assert p == pytest.approx(5.5735, abs=1e-3)


def test_put_call_parity() -> None:
    call = black_scholes.price(_S, _K, _T, _SIGMA, "C", _R)
    put = black_scholes.price(_S, _K, _T, _SIGMA, "P", _R)
    # C - P = S - K*e^{-rT}
    assert (call - put) == pytest.approx(_S - _K * math.exp(-_R * _T), abs=1e-6)


def test_call_greeks_match_reference() -> None:
    g = black_scholes.greeks(_S, _K, _T, _SIGMA, "C", _R)
    assert g.price == pytest.approx(10.4506, abs=1e-3)
    assert g.delta == pytest.approx(0.6368, abs=1e-3)
    assert g.gamma == pytest.approx(0.01876, abs=1e-4)
    assert g.vega == pytest.approx(0.37524, abs=1e-4)
    assert g.theta == pytest.approx(-0.017573, abs=1e-4)
    assert g.rho == pytest.approx(0.53232, abs=1e-3)


def test_put_greeks_match_reference() -> None:
    g = black_scholes.greeks(_S, _K, _T, _SIGMA, "P", _R)
    assert g.delta == pytest.approx(-0.3632, abs=1e-3)
    assert g.gamma == pytest.approx(0.01876, abs=1e-4)  # gamma is side-independent
    assert g.vega == pytest.approx(0.37524, abs=1e-4)  # vega is side-independent
    assert g.theta == pytest.approx(-0.004542, abs=1e-4)


def test_call_and_put_option_type_aliases_agree() -> None:
    assert black_scholes.price(_S, _K, _T, _SIGMA, "C", _R) == black_scholes.price(
        _S, _K, _T, _SIGMA, "CE", _R
    )
    assert black_scholes.price(_S, _K, _T, _SIGMA, "call", _R) == black_scholes.price(
        _S, _K, _T, _SIGMA, "CALL", _R
    )
    assert black_scholes.price(_S, _K, _T, _SIGMA, "P", _R) == black_scholes.price(
        _S, _K, _T, _SIGMA, "PE", _R
    )


def test_price_uses_module_default_risk_free_rate() -> None:
    assert black_scholes.price(_S, _K, _T, _SIGMA, "C") == black_scholes.price(
        _S, _K, _T, _SIGMA, "C", black_scholes.DEFAULT_RISK_FREE_RATE
    )
    assert pytest.approx(0.05, abs=1e-9) == black_scholes.DEFAULT_RISK_FREE_RATE


def test_price_at_expiry_or_zero_vol_is_intrinsic() -> None:
    # T=0: intrinsic value only, regardless of sigma.
    assert black_scholes.price(110.0, 100.0, 0.0, _SIGMA, "C", _R) == pytest.approx(10.0)
    assert black_scholes.price(90.0, 100.0, 0.0, _SIGMA, "C", _R) == pytest.approx(0.0)
    assert black_scholes.price(90.0, 100.0, 0.0, _SIGMA, "P", _R) == pytest.approx(10.0)
    # sigma=0 behaves the same way.
    assert black_scholes.price(110.0, 100.0, _T, 0.0, "C", _R) == pytest.approx(10.0)


def test_greeks_at_expiry_are_all_zero() -> None:
    g = black_scholes.greeks(110.0, 100.0, 0.0, _SIGMA, "C", _R)
    assert (g.delta, g.gamma, g.theta, g.vega, g.rho) == (0.0, 0.0, 0.0, 0.0, 0.0)
    assert g.price == pytest.approx(10.0)


def test_implied_vol_round_trips_price_to_iv() -> None:
    theo_price = black_scholes.price(_S, _K, _T, _SIGMA, "C", _R)
    iv = black_scholes.implied_vol(theo_price, _S, _K, _T, "C", _R)
    assert iv == pytest.approx(_SIGMA, abs=1e-4)


def test_implied_vol_round_trips_for_put_and_otm_strike() -> None:
    strike = 120.0
    sigma = 0.35
    theo_price = black_scholes.price(_S, strike, _T, sigma, "P", _R)
    iv = black_scholes.implied_vol(theo_price, _S, strike, _T, "P", _R)
    assert iv == pytest.approx(sigma, abs=1e-4)


def test_implied_vol_non_positive_price_is_nan() -> None:
    assert math.isnan(black_scholes.implied_vol(0.0, _S, _K, _T, "C", _R))
    assert math.isnan(black_scholes.implied_vol(-1.0, _S, _K, _T, "C", _R))


def test_implied_vol_zero_time_to_expiry_is_nan() -> None:
    assert math.isnan(black_scholes.implied_vol(5.0, _S, _K, 0.0, "C", _R))
