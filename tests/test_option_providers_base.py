"""StaticOptionChainProvider fixture behavior, nearest_expiry selection, and
the build_option_provider market factory."""

from __future__ import annotations

from datetime import UTC, date, datetime

from argus.markets import IN_NSE, US_NASDAQ, US_NYSE, Instrument
from argus.options.models import OptionChain, OptionQuote
from argus.options.providers.base import StaticOptionChainProvider, nearest_expiry
from argus.options.providers.factory import build_option_provider
from argus.options.providers.nse_options import NSEOptionsProvider
from argus.options.providers.yfinance_options import YFinanceOptionsProvider

_EXPIRY_1 = date(2026, 9, 25)
_EXPIRY_2 = date(2026, 10, 30)


def _chain() -> OptionChain:
    return OptionChain(
        symbol="AAPL",
        market_code="US_NASDAQ",
        spot=100.0,
        as_of=datetime(2026, 9, 1, tzinfo=UTC),
        expiries=[_EXPIRY_1, _EXPIRY_2],
        quotes=[OptionQuote(strike=100.0, expiry=_EXPIRY_1, right="C", last=3.0)],
    )


async def test_static_provider_supports_and_serves_added_chain() -> None:
    provider = StaticOptionChainProvider()
    inst = Instrument(symbol="AAPL", market_code="US_NASDAQ")
    assert provider.supports(inst) is False

    provider.add(_chain())
    assert provider.supports(inst) is True

    chain = await provider.get_chain(inst)
    assert chain is not None
    assert chain.symbol == "AAPL"

    expiries = await provider.list_expiries(inst)
    assert expiries == [_EXPIRY_1, _EXPIRY_2]


async def test_static_provider_get_chain_unknown_expiry_returns_none() -> None:
    provider = StaticOptionChainProvider()
    provider.add(_chain())
    inst = Instrument(symbol="AAPL", market_code="US_NASDAQ")
    assert await provider.get_chain(inst, date(2099, 1, 1)) is None


async def test_static_provider_unknown_instrument_returns_none() -> None:
    provider = StaticOptionChainProvider()
    inst = Instrument(symbol="MSFT", market_code="US_NASDAQ")
    assert await provider.get_chain(inst) is None
    assert await provider.list_expiries(inst) == []


async def test_static_provider_health_check_and_aclose() -> None:
    provider = StaticOptionChainProvider()
    health = await provider.health_check()
    assert health.ok is True
    await provider.aclose()  # must not raise


def test_nearest_expiry_picks_soonest_on_or_after_today() -> None:
    today = date(2026, 9, 10)
    expiries = [date(2026, 8, 1), date(2026, 9, 25), date(2026, 10, 30)]
    assert nearest_expiry(expiries, today=today) == date(2026, 9, 25)


def test_nearest_expiry_all_past_falls_back_to_earliest() -> None:
    today = date(2027, 1, 1)
    expiries = [date(2026, 8, 1), date(2026, 9, 25)]
    assert nearest_expiry(expiries, today=today) == date(2026, 8, 1)


def test_nearest_expiry_empty_is_none() -> None:
    assert nearest_expiry([]) is None


def test_build_option_provider_routes_by_market() -> None:
    nse = build_option_provider(Instrument(symbol="RELIANCE", market_code=IN_NSE.code))
    assert isinstance(nse, NSEOptionsProvider)

    nasdaq = build_option_provider(Instrument(symbol="AAPL", market_code=US_NASDAQ.code))
    assert isinstance(nasdaq, YFinanceOptionsProvider)

    nyse = build_option_provider(Instrument(symbol="JPM", market_code=US_NYSE.code))
    assert isinstance(nyse, YFinanceOptionsProvider)


def test_build_option_provider_unsupported_market_returns_none() -> None:
    assert build_option_provider(Instrument(symbol="X", market_code="UNKNOWN")) is None
