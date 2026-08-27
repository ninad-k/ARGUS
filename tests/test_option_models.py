"""OptionChain helpers: for_expiry, calls/puts, strikes, atm_strike, mid fallback."""

from __future__ import annotations

from datetime import UTC, date, datetime

from argus.options.models import OptionChain, OptionQuote

_EXPIRY_1 = date(2026, 9, 25)
_EXPIRY_2 = date(2026, 10, 30)


def _chain() -> OptionChain:
    quotes = [
        OptionQuote(strike=95.0, expiry=_EXPIRY_1, right="C", bid=6.0, ask=6.4, last=6.1),
        OptionQuote(strike=100.0, expiry=_EXPIRY_1, right="C", bid=3.0, ask=3.2, last=3.1),
        OptionQuote(strike=105.0, expiry=_EXPIRY_1, right="C", bid=None, ask=None, last=1.2),
        OptionQuote(strike=95.0, expiry=_EXPIRY_1, right="P", bid=1.0, ask=1.2, last=1.1),
        OptionQuote(strike=100.0, expiry=_EXPIRY_1, right="P", bid=2.9, ask=3.1, last=3.0),
        OptionQuote(strike=105.0, expiry=_EXPIRY_1, right="P", bid=None, ask=None, last=None),
        # A second expiry, to prove for_expiry/strikes don't leak across expiries.
        OptionQuote(strike=100.0, expiry=_EXPIRY_2, right="C", bid=5.0, ask=5.4, last=5.2),
    ]
    return OptionChain(
        symbol="TEST",
        market_code="US_NASDAQ",
        spot=101.0,
        as_of=datetime(2026, 9, 1, tzinfo=UTC),
        expiries=[_EXPIRY_1, _EXPIRY_2],
        quotes=quotes,
    )


def test_for_expiry_filters_to_that_expiry_only() -> None:
    chain = _chain()
    rows = chain.for_expiry(_EXPIRY_1)
    assert len(rows) == 6
    assert all(q.expiry == _EXPIRY_1 for q in rows)


def test_calls_and_puts_split_by_right() -> None:
    chain = _chain()
    assert {q.strike for q in chain.calls(_EXPIRY_1)} == {95.0, 100.0, 105.0}
    assert {q.strike for q in chain.puts(_EXPIRY_1)} == {95.0, 100.0, 105.0}
    assert all(q.right == "C" for q in chain.calls(_EXPIRY_1))
    assert all(q.right == "P" for q in chain.puts(_EXPIRY_1))


def test_strikes_sorted_and_deduplicated() -> None:
    chain = _chain()
    assert chain.strikes(_EXPIRY_1) == [95.0, 100.0, 105.0]


def test_atm_strike_is_closest_to_spot() -> None:
    chain = _chain()
    # spot=101.0 -- closer to 100.0 than to 95.0 or 105.0.
    assert chain.atm_strike(_EXPIRY_1) == 100.0


def test_atm_strike_none_when_expiry_has_no_quotes() -> None:
    chain = _chain()
    assert chain.atm_strike(date(2027, 1, 1)) is None


def test_mid_uses_bid_ask_midpoint_when_both_present() -> None:
    quote = OptionQuote(strike=100.0, expiry=_EXPIRY_1, right="C", bid=3.0, ask=3.2, last=99.0)
    assert OptionChain.mid(quote) == 3.1


def test_mid_falls_back_to_last_when_bid_or_ask_missing() -> None:
    only_ask = OptionQuote(strike=100.0, expiry=_EXPIRY_1, right="C", bid=None, ask=3.2, last=3.1)
    only_bid = OptionQuote(strike=100.0, expiry=_EXPIRY_1, right="C", bid=3.0, ask=None, last=3.1)
    neither = OptionQuote(strike=100.0, expiry=_EXPIRY_1, right="C", bid=None, ask=None, last=3.1)
    assert OptionChain.mid(only_ask) == 3.1
    assert OptionChain.mid(only_bid) == 3.1
    assert OptionChain.mid(neither) == 3.1


def test_mid_none_when_nothing_available() -> None:
    quote = OptionQuote(strike=100.0, expiry=_EXPIRY_1, right="C", bid=None, ask=None, last=None)
    assert OptionChain.mid(quote) is None
