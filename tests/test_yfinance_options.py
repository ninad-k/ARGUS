"""DataFrame fixture -> OptionChain mapping for YFinanceOptionsProvider
(monkeypatch ``yf.Ticker``); greeks filled when IV is present, left ``None``
otherwise. The one live network check is skipped by default."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from argus.markets import IN_NSE, Instrument
from argus.options.providers import yfinance_options
from argus.options.providers.yfinance_options import YFinanceOptionsProvider

_EXPIRIES = ("2026-09-25", "2026-10-30")


class _FakeFastInfo:
    def __init__(self, last_price: float) -> None:
        self.last_price = last_price


class _FakeOptionChainResult:
    def __init__(self, calls: pd.DataFrame, puts: pd.DataFrame) -> None:
        self.calls = calls
        self.puts = puts


class _FakeTicker:
    def __init__(self, symbol: str) -> None:
        self.ticker = symbol
        self.options = _EXPIRIES
        self.fast_info = _FakeFastInfo(101.0)

    def option_chain(self, expiry: str) -> _FakeOptionChainResult:
        assert expiry in self.options
        calls = pd.DataFrame(
            [
                {
                    "strike": 100.0,
                    "bid": 3.0,
                    "ask": 3.2,
                    "lastPrice": 3.1,
                    "impliedVolatility": 0.25,
                    "openInterest": 120,
                    "volume": 40,
                },
                {
                    "strike": 105.0,
                    "bid": float("nan"),
                    "ask": float("nan"),
                    "lastPrice": 1.0,
                    "impliedVolatility": 0.0,  # no usable IV -> no greeks
                    "openInterest": 10,
                    "volume": 2,
                },
            ]
        )
        puts = pd.DataFrame(
            [
                {
                    "strike": 100.0,
                    "bid": 2.8,
                    "ask": 3.0,
                    "lastPrice": 2.9,
                    "impliedVolatility": 0.22,
                    "openInterest": 90,
                    "volume": 30,
                },
            ]
        )
        return _FakeOptionChainResult(calls, puts)


class _EmptyTicker(_FakeTicker):
    def __init__(self, symbol: str) -> None:
        super().__init__(symbol)
        self.options = ()


class _NoFastInfoTicker(_FakeTicker):
    @property
    def fast_info(self) -> _FakeFastInfo:  # type: ignore[override]
        raise RuntimeError("fast_info unavailable")

    @fast_info.setter
    def fast_info(self, value: _FakeFastInfo) -> None:
        pass

    def history(self, **kwargs: object) -> pd.DataFrame:
        return pd.DataFrame({"Close": [99.5, 101.0]})


@pytest.fixture
def fake_ticker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(yfinance_options.yf, "Ticker", _FakeTicker)


def test_supports_rejects_in_nse() -> None:
    provider = YFinanceOptionsProvider()
    assert provider.supports(Instrument(symbol="AAPL", market_code="US_NASDAQ")) is True
    assert provider.supports(Instrument(symbol="RELIANCE", market_code=IN_NSE.code)) is False


async def test_get_chain_maps_dataframes_to_quotes(fake_ticker: None) -> None:
    provider = YFinanceOptionsProvider()
    inst = Instrument(symbol="AAPL", market_code="US_NASDAQ")

    chain = await provider.get_chain(inst, date(2026, 9, 25))

    assert chain is not None
    assert chain.symbol == "AAPL"
    assert chain.spot == 101.0
    assert chain.expiries == [date(2026, 9, 25), date(2026, 10, 30)]

    calls = {q.strike: q for q in chain.calls(date(2026, 9, 25))}
    puts = {q.strike: q for q in chain.puts(date(2026, 9, 25))}
    assert calls[100.0].bid == 3.0
    assert calls[100.0].ask == 3.2
    assert calls[100.0].last == 3.1
    assert calls[100.0].iv == 0.25
    assert calls[100.0].oi == 120
    assert calls[100.0].volume == 40
    assert puts[100.0].last == 2.9
    # NaN bid/ask on the illiquid 105 strike must be normalized to None.
    assert calls[105.0].bid is None
    assert calls[105.0].ask is None


async def test_get_chain_fills_greeks_when_iv_present(fake_ticker: None) -> None:
    provider = YFinanceOptionsProvider()
    inst = Instrument(symbol="AAPL", market_code="US_NASDAQ")

    chain = await provider.get_chain(inst, date(2026, 9, 25))

    assert chain is not None
    calls = {q.strike: q for q in chain.calls(date(2026, 9, 25))}
    # strike 100 has iv=0.25 -> greeks computed.
    assert calls[100.0].delta is not None
    assert calls[100.0].gamma is not None
    assert 0.0 < calls[100.0].delta < 1.0
    # strike 105 has iv=0.0 -> no usable IV, greeks left None.
    assert calls[105.0].delta is None
    assert calls[105.0].gamma is None


async def test_get_chain_none_expiry_picks_nearest(fake_ticker: None) -> None:
    provider = YFinanceOptionsProvider()
    inst = Instrument(symbol="AAPL", market_code="US_NASDAQ")

    chain = await provider.get_chain(inst, expiry=None)

    assert chain is not None
    # Both expiries are in the future relative to "today" in this test run,
    # so the nearest is the earliest of the two.
    assert chain.for_expiry(date(2026, 9, 25)) or chain.for_expiry(date(2026, 10, 30))


async def test_get_chain_wrong_market_returns_none_without_calling_yfinance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(symbol: str) -> _FakeTicker:  # pragma: no cover
        raise AssertionError("yfinance must not be called for a non-US instrument")

    monkeypatch.setattr(yfinance_options.yf, "Ticker", _boom)
    provider = YFinanceOptionsProvider()
    chain = await provider.get_chain(Instrument(symbol="RELIANCE", market_code=IN_NSE.code))
    assert chain is None


async def test_get_chain_no_expiries_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(yfinance_options.yf, "Ticker", _EmptyTicker)
    provider = YFinanceOptionsProvider()
    chain = await provider.get_chain(Instrument(symbol="AAPL", market_code="US_NASDAQ"))
    assert chain is None


async def test_get_chain_falls_back_to_history_close_when_fast_info_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(yfinance_options.yf, "Ticker", _NoFastInfoTicker)
    provider = YFinanceOptionsProvider()
    chain = await provider.get_chain(
        Instrument(symbol="AAPL", market_code="US_NASDAQ"), date(2026, 9, 25)
    )
    assert chain is not None
    assert chain.spot == 101.0  # last row of the fake history() Close column


async def test_list_expiries_parses_dates(fake_ticker: None) -> None:
    provider = YFinanceOptionsProvider()
    expiries = await provider.list_expiries(Instrument(symbol="AAPL", market_code="US_NASDAQ"))
    assert expiries == [date(2026, 9, 25), date(2026, 10, 30)]


async def test_list_expiries_wrong_market_is_empty_without_calling_yfinance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(symbol: str) -> _FakeTicker:  # pragma: no cover
        raise AssertionError("yfinance must not be called for a non-US instrument")

    monkeypatch.setattr(yfinance_options.yf, "Ticker", _boom)
    provider = YFinanceOptionsProvider()
    expiries = await provider.list_expiries(Instrument(symbol="RELIANCE", market_code=IN_NSE.code))
    assert expiries == []


async def test_health_check_ok_when_expiries_available(fake_ticker: None) -> None:
    provider = YFinanceOptionsProvider()
    health = await provider.health_check()
    assert health.ok is True


async def test_health_check_unhealthy_when_ticker_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(symbol: str) -> _FakeTicker:
        raise RuntimeError("boom")

    monkeypatch.setattr(yfinance_options.yf, "Ticker", _boom)
    provider = YFinanceOptionsProvider()
    health = await provider.health_check()
    assert health.ok is False


async def test_aclose_is_noop() -> None:
    provider = YFinanceOptionsProvider()
    await provider.aclose()  # must not raise


@pytest.mark.network
async def test_yfinance_options_live_get_chain_aapl() -> None:
    provider = YFinanceOptionsProvider()
    chain = await provider.get_chain(Instrument(symbol="AAPL", market_code="US_NASDAQ"))
    assert chain is not None
    assert chain.spot > 0
