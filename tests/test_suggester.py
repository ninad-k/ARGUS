"""``argus.options.suggester``: delta-band selection per risk level, liquidity
filtering (incl. the thin-liquidity relax path), the IV-Rank "expensive
premium" guard (deeper-ITM preference / futures preference), expiry-window
selection, the short/PE path, and the ``None``-returning edge cases."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from argus.config.options import OptionsSettings
from argus.markets import US_NASDAQ, Instrument
from argus.options.models import OptionChain, OptionQuote
from argus.options.suggester import (
    DerivativeSuggestion,
    RiskLevel,
    select_expiry,
    suggest_derivative,
)
from argus.screener.base import Candidate

_TODAY = date.today()  # noqa: DTZ011 -- matches suggester's own DTE-boundary convention
_EXPIRY = _TODAY + timedelta(days=45)  # inside the default 20-60 day window


def _settings(**overrides: object) -> OptionsSettings:
    return OptionsSettings(_env_file=None, **overrides)  # type: ignore[arg-type]


def _candidate(
    symbol: str = "NVDA",
    *,
    direction: str = "long",
    has_options: bool = True,
    has_futures: bool = False,
    lot_size: int = 1,
) -> Candidate:
    inst = Instrument(
        symbol=symbol,
        market_code=US_NASDAQ.code,
        has_options=has_options,
        has_futures=has_futures,
        lot_size=lot_size,
    )
    return Candidate(instrument=inst, strategy="momentum", score=80.0, direction=direction)  # type: ignore[arg-type]


def _quote(
    strike: float,
    right: str = "C",
    *,
    delta: float | None = None,
    oi: float | None = 500.0,
    bid: float | None = 9.5,
    ask: float | None = 10.5,
    last: float | None = 10.0,
    iv: float | None = 0.30,
    expiry: date = _EXPIRY,
) -> OptionQuote:
    return OptionQuote(
        strike=strike,
        expiry=expiry,
        right=right,  # type: ignore[arg-type]
        bid=bid,
        ask=ask,
        last=last,
        iv=iv,
        oi=oi,
        delta=delta,
    )


def _chain(
    quotes: list[OptionQuote], *, spot: float = 120.0, expiries: list[date] | None = None
) -> OptionChain:
    return OptionChain(
        symbol="NVDA",
        market_code=US_NASDAQ.code,
        spot=spot,
        as_of=datetime.now(UTC),
        expiries=expiries if expiries is not None else [_EXPIRY],
        quotes=quotes,
    )


# --- delta-band selection per risk level --------------------------------


def test_delta_band_selection_spans_conservative_moderate_aggressive() -> None:
    """Strikes span Δ 0.10-0.80; each risk level should pick the strike
    whose delta falls in its own band."""
    quotes = [
        _quote(100.0, delta=0.80),  # too deep ITM for any band
        _quote(110.0, delta=0.65),  # conservative band (0.55-0.70)
        _quote(120.0, delta=0.45),  # moderate band (0.35-0.50)
        _quote(130.0, delta=0.25),  # aggressive band (0.15-0.30)
        _quote(140.0, delta=0.10),  # too far OTM for any band
    ]
    chain = _chain(quotes)
    candidate = _candidate()
    settings = _settings()

    conservative = suggest_derivative(
        candidate, chain, risk=RiskLevel.CONSERVATIVE, settings=settings
    )
    moderate = suggest_derivative(candidate, chain, risk=RiskLevel.MODERATE, settings=settings)
    aggressive = suggest_derivative(
        candidate, chain, risk=RiskLevel.AGGRESSIVE, settings=settings
    )

    assert conservative is not None and conservative.strike == 110.0
    assert moderate is not None and moderate.strike == 120.0
    assert aggressive is not None and aggressive.strike == 130.0
    assert conservative.instrument_type == "CE"
    assert "CONSERVATIVE" in conservative.rationale
    assert "NVDA" in conservative.rationale


# --- liquidity filter -----------------------------------------------------


def test_liquidity_filter_excludes_low_oi() -> None:
    quotes = [
        _quote(110.0, delta=0.60, oi=50.0),  # below min_oi=100 -- excluded
        _quote(115.0, delta=0.62, oi=500.0),
    ]
    chain = _chain(quotes)
    result = suggest_derivative(
        _candidate(), chain, risk=RiskLevel.CONSERVATIVE, settings=_settings()
    )
    assert result is not None
    assert result.strike == 115.0


def test_liquidity_filter_excludes_wide_spread() -> None:
    quotes = [
        # spread = (12-8)/10 = 40% > 10% max -- excluded
        _quote(110.0, delta=0.60, oi=500.0, bid=8.0, ask=12.0),
        _quote(115.0, delta=0.62, oi=500.0, bid=9.5, ask=10.5),
    ]
    chain = _chain(quotes)
    result = suggest_derivative(
        _candidate(), chain, risk=RiskLevel.CONSERVATIVE, settings=_settings()
    )
    assert result is not None
    assert result.strike == 115.0


def test_liquidity_filter_relaxes_to_last_when_nothing_passes() -> None:
    quotes = [
        _quote(110.0, delta=0.60, oi=10.0, bid=None, ask=None, last=11.0),
    ]
    chain = _chain(quotes)
    result = suggest_derivative(
        _candidate(), chain, risk=RiskLevel.CONSERVATIVE, settings=_settings()
    )
    assert result is not None
    assert result.strike == 110.0
    assert result.suggested_price == 11.0
    assert "thin liquidity" in result.rationale


def test_liquidity_filter_returns_none_when_nothing_at_all_passes() -> None:
    quotes = [
        _quote(110.0, delta=0.60, oi=10.0, bid=None, ask=None, last=None),
    ]
    chain = _chain(quotes)
    result = suggest_derivative(
        _candidate(), chain, risk=RiskLevel.CONSERVATIVE, settings=_settings()
    )
    assert result is None


# --- IV-Rank "expensive premium" guard ------------------------------------


def test_high_ivr_conservative_with_futures_prefers_futures() -> None:
    quotes = [
        _quote(110.0, delta=0.60, iv=0.90),
    ]
    chain = _chain(quotes, spot=112.0)
    candidate = _candidate(has_futures=True)
    # iv_history low relative to the current ATM iv (0.90) -> IVR ~100
    result = suggest_derivative(
        candidate,
        chain,
        risk=RiskLevel.CONSERVATIVE,
        settings=_settings(),
        iv_history=[0.10, 0.20, 0.30, 0.40, 0.50],
    )
    assert result is not None
    assert result.instrument_type == "FUT"
    assert result.strike is None
    assert result.suggested_price == 112.0
    assert result.est_cost == 112.0 * 1 * 0.15
    assert "IVR" in result.rationale


def test_high_ivr_moderate_without_futures_prefers_deeper_itm() -> None:
    quotes = [
        _quote(115.0, delta=0.36, iv=0.90),  # low end of moderate band
        _quote(110.0, delta=0.49, iv=0.90),  # high end -> deeper ITM
    ]
    chain = _chain(quotes, spot=112.0)
    candidate = _candidate(has_futures=False)
    result = suggest_derivative(
        candidate,
        chain,
        risk=RiskLevel.MODERATE,
        settings=_settings(),
        iv_history=[0.10, 0.20, 0.30, 0.40, 0.50],
    )
    assert result is not None
    assert result.instrument_type == "CE"
    assert result.strike == 110.0  # deepest |delta| among liquid, in-band quotes
    assert "elevated IV" in result.rationale


def test_high_ivr_aggressive_still_allowed_with_note_and_uses_normal_midpoint() -> None:
    """Unlike conservative/moderate, aggressive keeps the usual
    closest-to-band-midpoint pick even when IV is elevated -- it only adds a
    rationale note, it doesn't switch to the deepest-ITM strike."""
    quotes = [
        _quote(135.0, delta=0.15, iv=0.90),  # band edge, not the midpoint
        _quote(130.0, delta=0.225, iv=0.90),  # closest to band midpoint (0.225)
        _quote(125.0, delta=0.30, iv=0.90),  # deepest ITM in band -- must NOT be chosen
    ]
    chain = _chain(quotes, spot=112.0)
    candidate = _candidate(has_futures=True)  # has_futures, but risk is aggressive not conservative
    result = suggest_derivative(
        candidate,
        chain,
        risk=RiskLevel.AGGRESSIVE,
        settings=_settings(),
        iv_history=[0.10, 0.20, 0.30, 0.40, 0.50],
    )
    assert result is not None
    assert result.instrument_type == "CE"
    assert result.strike == 130.0
    assert "elevated IV" in result.rationale
    assert "deeper ITM" not in result.rationale


def test_empty_iv_history_skips_ivr_guard_entirely() -> None:
    """No history -> IVR guard is skipped, even with has_futures + conservative."""
    quotes = [
        _quote(110.0, delta=0.60, iv=0.90),
    ]
    chain = _chain(quotes, spot=112.0)
    candidate = _candidate(has_futures=True)
    result = suggest_derivative(
        candidate, chain, risk=RiskLevel.CONSERVATIVE, settings=_settings(), iv_history=()
    )
    assert result is not None
    assert result.instrument_type == "CE"  # not FUT -- guard never fired


# --- expiry selection ------------------------------------------------------


def test_select_expiry_prefers_nearest_within_window() -> None:
    expiries = [
        _TODAY + timedelta(days=10),
        _TODAY + timedelta(days=25),
        _TODAY + timedelta(days=40),
        _TODAY + timedelta(days=90),
    ]
    chain = _chain([], expiries=expiries)
    chosen = select_expiry(chain, min_days=20, max_days=60)
    assert chosen == _TODAY + timedelta(days=25)


def test_select_expiry_falls_back_to_nearest_at_least_min_days() -> None:
    expiries = [_TODAY + timedelta(days=5), _TODAY + timedelta(days=90)]
    chain = _chain([], expiries=expiries)
    chosen = select_expiry(chain, min_days=20, max_days=60)
    assert chosen == _TODAY + timedelta(days=90)


def test_select_expiry_falls_back_to_nearest_available_when_all_too_soon() -> None:
    expiries = [_TODAY + timedelta(days=3), _TODAY + timedelta(days=7)]
    chain = _chain([], expiries=expiries)
    chosen = select_expiry(chain, min_days=20, max_days=60)
    assert chosen == _TODAY + timedelta(days=7)


def test_select_expiry_returns_none_when_no_expiries() -> None:
    chain = _chain([], expiries=[])
    assert select_expiry(chain) is None


# --- short direction / PE path ---------------------------------------------


def test_short_direction_selects_puts() -> None:
    quotes = [
        _quote(110.0, right="P", delta=-0.60, oi=500.0),
        _quote(130.0, right="C", delta=0.60, oi=500.0),  # calls present but ignored
    ]
    chain = _chain(quotes)
    candidate = _candidate(direction="short")
    result = suggest_derivative(candidate, chain, risk=RiskLevel.CONSERVATIVE, settings=_settings())
    assert result is not None
    assert result.instrument_type == "PE"
    assert result.strike == 110.0


# --- None-returning edge cases ----------------------------------------------


def test_returns_none_when_no_expiries_at_all() -> None:
    chain = _chain([], expiries=[])
    result = suggest_derivative(_candidate(), chain, risk=RiskLevel.MODERATE, settings=_settings())
    assert result is None


def test_returns_none_when_no_quotes_have_deltas() -> None:
    quotes = [_quote(110.0, delta=None)]
    chain = _chain(quotes)
    result = suggest_derivative(_candidate(), chain, risk=RiskLevel.MODERATE, settings=_settings())
    assert result is None


def test_returns_none_when_instrument_has_no_derivatives() -> None:
    quotes = [_quote(110.0, delta=0.60)]
    chain = _chain(quotes)
    candidate = _candidate(has_options=False, has_futures=False)
    result = suggest_derivative(candidate, chain, risk=RiskLevel.MODERATE, settings=_settings())
    assert result is None


def test_est_cost_scales_with_lot_size() -> None:
    quotes = [_quote(110.0, delta=0.60, bid=9.5, ask=10.5)]
    chain = _chain(quotes)
    candidate = _candidate(lot_size=75)
    result = suggest_derivative(candidate, chain, risk=RiskLevel.CONSERVATIVE, settings=_settings())
    assert result is not None
    assert result.suggested_price == 10.0
    assert result.est_cost == 10.0 * 75
    assert isinstance(result, DerivativeSuggestion)
