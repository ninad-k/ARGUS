"""max_pain / PCR on a small hand-built chain with a known answer; GEX sign
convention; iv_rank edge cases."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from argus.options.analytics import atm_iv, gex_profile, iv_rank, max_pain, oi_profile, pcr
from argus.options.models import OptionChain, OptionQuote

_EXPIRY = date(2026, 9, 25)


def _quote(
    strike: float,
    right: str,
    oi: float,
    volume: float,
    gamma: float | None = None,
    iv: float | None = None,
) -> OptionQuote:
    return OptionQuote(
        strike=strike,
        expiry=_EXPIRY,
        right=right,  # type: ignore[arg-type]
        oi=oi,
        volume=volume,
        gamma=gamma,
        iv=iv,
    )


def _known_chain() -> OptionChain:
    """Strikes 95/100/105. OI/volume chosen so max_pain has a unique minimum
    at strike=100 (see module-level comment math below) and PCR has a clean
    non-1.0 answer.

    payout(95)  = calls: (95-95)*100=0
                + puts:  (95-95)*300 + (100-95)*150 + (105-95)*50 = 1250
                = 1250
    payout(100) = calls: (100-95)*100 + (100-100)*200 = 500
                + puts:  (100-100)*150 + (105-100)*50 = 250
                = 750   <- minimum
    payout(105) = calls: (105-95)*100 + (105-100)*200 + (105-105)*50 = 2000
                + puts:  (105-105)*50 = 0
                = 2000
    """
    quotes = [
        _quote(95.0, "C", oi=100, volume=40, gamma=0.02),
        _quote(100.0, "C", oi=200, volume=80, gamma=0.02),
        _quote(105.0, "C", oi=50, volume=20, gamma=0.02),
        _quote(95.0, "P", oi=300, volume=120, gamma=0.025),
        _quote(100.0, "P", oi=150, volume=60, gamma=0.025),
        _quote(105.0, "P", oi=50, volume=30, gamma=0.025),
    ]
    return OptionChain(
        symbol="TEST",
        market_code="US_NASDAQ",
        spot=100.0,
        as_of=datetime(2026, 9, 1, tzinfo=UTC),
        expiries=[_EXPIRY],
        quotes=quotes,
    )


def test_oi_profile_per_strike_totals() -> None:
    profile = oi_profile(_known_chain(), _EXPIRY)
    by_strike = {p.strike: p for p in profile}
    assert by_strike[95.0].call_oi == 100
    assert by_strike[95.0].put_oi == 300
    assert by_strike[100.0].call_volume == 80
    assert by_strike[100.0].put_volume == 60


def test_oi_profile_missing_side_treated_as_zero() -> None:
    chain = OptionChain(
        symbol="TEST",
        market_code="US_NASDAQ",
        spot=100.0,
        as_of=datetime(2026, 9, 1, tzinfo=UTC),
        expiries=[_EXPIRY],
        quotes=[_quote(100.0, "C", oi=10, volume=1)],
    )
    profile = oi_profile(chain, _EXPIRY)
    assert len(profile) == 1
    assert profile[0].put_oi == 0.0
    assert profile[0].put_volume == 0.0


def test_max_pain_is_the_minimum_payout_strike() -> None:
    assert max_pain(_known_chain(), _EXPIRY) == 100.0


def test_max_pain_empty_chain_falls_back_to_spot() -> None:
    chain = OptionChain(
        symbol="TEST",
        market_code="US_NASDAQ",
        spot=42.0,
        as_of=datetime(2026, 9, 1, tzinfo=UTC),
        expiries=[],
        quotes=[],
    )
    assert max_pain(chain, _EXPIRY) == 42.0


def test_pcr_by_oi_and_volume() -> None:
    pcr_oi, pcr_volume = pcr(_known_chain(), _EXPIRY)
    # total_call_oi=350, total_put_oi=500 -> 500/350
    assert pcr_oi == pytest.approx(500.0 / 350.0)
    # total_call_vol=140, total_put_vol=210 -> 210/140
    assert pcr_volume == pytest.approx(210.0 / 140.0)


def test_pcr_zero_call_side_is_zero_not_a_crash() -> None:
    chain = OptionChain(
        symbol="TEST",
        market_code="US_NASDAQ",
        spot=100.0,
        as_of=datetime(2026, 9, 1, tzinfo=UTC),
        expiries=[_EXPIRY],
        quotes=[_quote(100.0, "P", oi=10, volume=5)],
    )
    assert pcr(chain, _EXPIRY) == (0.0, 0.0)


def test_gex_profile_matches_expected_values() -> None:
    profile = dict(gex_profile(_known_chain(), _EXPIRY, spot=100.0))
    # strike95: (-100*0.02 + 300*0.025) * 100^2*0.01 = 5.5 * 100 = 550
    assert profile[95.0] == pytest.approx(550.0)
    # strike100: (-200*0.02 + 150*0.025) * 100 = -0.25 * 100 = -25
    assert profile[100.0] == pytest.approx(-25.0)
    # strike105: (-50*0.02 + 50*0.025) * 100 = 0.25 * 100 = 25
    assert profile[105.0] == pytest.approx(25.0)


def test_gex_sign_convention_calls_negative_puts_positive() -> None:
    """DRUVA's dealer-hedge convention (dealers short calls & short puts):
    gex = (-call_oi*call_gamma + put_oi*put_gamma) * spot^2 * 0.01 -- so an
    all-call strike contributes negative GEX and an all-put strike
    contributes positive GEX (both gammas are non-negative; the sign comes
    from which side has the OI)."""
    call_only = OptionChain(
        symbol="CALLONLY",
        market_code="US_NASDAQ",
        spot=100.0,
        as_of=datetime(2026, 9, 1, tzinfo=UTC),
        expiries=[_EXPIRY],
        quotes=[_quote(100.0, "C", oi=100, volume=1, gamma=0.02)],
    )
    put_only = OptionChain(
        symbol="PUTONLY",
        market_code="US_NASDAQ",
        spot=100.0,
        as_of=datetime(2026, 9, 1, tzinfo=UTC),
        expiries=[_EXPIRY],
        quotes=[_quote(100.0, "P", oi=100, volume=1, gamma=0.02)],
    )
    [(_, call_gex)] = gex_profile(call_only, _EXPIRY, spot=100.0)
    [(_, put_gex)] = gex_profile(put_only, _EXPIRY, spot=100.0)
    assert call_gex < 0
    assert put_gex > 0
    assert call_gex == pytest.approx(-put_gex)


def test_iv_rank_empty_history_is_zero() -> None:
    assert iv_rank(0.5, []) == 0.0


def test_iv_rank_current_above_all_history_clamps_to_100() -> None:
    assert iv_rank(1.0, [0.1, 0.2, 0.3]) == 100.0


def test_iv_rank_current_below_all_history_clamps_to_0() -> None:
    assert iv_rank(0.01, [0.1, 0.2, 0.3]) == 0.0


def test_iv_rank_midrange_value() -> None:
    assert iv_rank(0.25, [0.1, 0.2, 0.3, 0.4]) == pytest.approx(50.0)


def test_iv_rank_zero_range_history_is_zero() -> None:
    assert iv_rank(0.2, [0.2, 0.2, 0.2]) == 0.0


def test_atm_iv_averages_call_and_put_at_atm_strike() -> None:
    chain = OptionChain(
        symbol="TEST",
        market_code="US_NASDAQ",
        spot=100.0,
        as_of=datetime(2026, 9, 1, tzinfo=UTC),
        expiries=[_EXPIRY],
        quotes=[
            _quote(100.0, "C", oi=1, volume=1, iv=0.20),
            _quote(100.0, "P", oi=1, volume=1, iv=0.24),
            _quote(105.0, "C", oi=1, volume=1, iv=0.99),  # not ATM -- must be excluded
        ],
    )
    assert atm_iv(chain, _EXPIRY) == pytest.approx(0.22)


def test_atm_iv_none_when_no_iv_available() -> None:
    chain = OptionChain(
        symbol="TEST",
        market_code="US_NASDAQ",
        spot=100.0,
        as_of=datetime(2026, 9, 1, tzinfo=UTC),
        expiries=[_EXPIRY],
        quotes=[_quote(100.0, "C", oi=1, volume=1, iv=None)],
    )
    assert atm_iv(chain, _EXPIRY) is None


def test_atm_iv_none_when_expiry_has_no_quotes() -> None:
    chain = OptionChain(
        symbol="TEST",
        market_code="US_NASDAQ",
        spot=100.0,
        as_of=datetime(2026, 9, 1, tzinfo=UTC),
        expiries=[],
        quotes=[],
    )
    assert atm_iv(chain, _EXPIRY) is None
