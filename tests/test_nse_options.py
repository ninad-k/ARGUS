"""Canned NSE option-chain JSON -> chain parse (percent-IV normalization,
"27-Aug-2026"-style expiry parsing), warm-up + 403 tolerance via
``httpx.MockTransport``. The one live check is network-marked and skipped
by default (see pytest addopts)."""

from __future__ import annotations

from datetime import date

import httpx
import pytest

from argus.markets import IN_NSE, US_NASDAQ, Instrument
from argus.options.providers.nse_options import NSEOptionsProvider


def _leg(*, last: float, bid: float, ask: float, oi: int, volume: int, iv_pct: float) -> dict:
    return {
        "lastPrice": last,
        "bidprice": bid,
        "askPrice": ask,
        "openInterest": oi,
        "totalTradedVolume": volume,
        "impliedVolatility": iv_pct,
    }


def _payload(symbol: str = "RELIANCE") -> dict:
    return {
        "records": {
            "underlyingValue": 1300.0,
            "data": [
                {
                    "strikePrice": 1280,
                    "expiryDate": "27-Aug-2026",
                    "CE": _leg(last=35.0, bid=34.5, ask=35.5, oi=1000, volume=200, iv_pct=22.5),
                    "PE": _leg(last=12.0, bid=11.5, ask=12.5, oi=1500, volume=300, iv_pct=24.0),
                },
                {
                    "strikePrice": 1300,
                    "expiryDate": "27-Aug-2026",
                    "CE": _leg(last=20.0, bid=19.5, ask=20.5, oi=2000, volume=500, iv_pct=21.0),
                    "PE": _leg(last=18.0, bid=17.5, ask=18.5, oi=1800, volume=400, iv_pct=21.5),
                },
                {
                    "strikePrice": 1300,
                    "expiryDate": "24-Sep-2026",
                    "CE": _leg(last=30.0, bid=29.5, ask=30.5, oi=800, volume=100, iv_pct=0.0),
                    "PE": _leg(last=28.0, bid=27.5, ask=28.5, oi=700, volume=90, iv_pct=19.0),
                },
            ],
        }
    }


def _handler_for(symbol: str, path: str):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/":
            return httpx.Response(200, text="<html></html>")
        if request.url.path == path:
            assert request.url.params.get("symbol") == symbol
            return httpx.Response(200, json=_payload(symbol))
        return httpx.Response(404)  # pragma: no cover

    return handler


def test_supports_only_in_nse() -> None:
    provider = NSEOptionsProvider()
    assert provider.supports(Instrument(symbol="RELIANCE", market_code=IN_NSE.code)) is True
    assert provider.supports(Instrument(symbol="AAPL", market_code=US_NASDAQ.code)) is False


async def test_get_chain_wrong_market_returns_none_without_http_call() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("no HTTP call expected for a non-IN_NSE instrument")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = NSEOptionsProvider(timeout_seconds=5.0, http=client)
        chain = await provider.get_chain(Instrument(symbol="AAPL", market_code=US_NASDAQ.code))

    assert chain is None


async def test_get_chain_routes_stock_symbol_to_equities_endpoint() -> None:
    handler = _handler_for("RELIANCE", "/api/option-chain-equities")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = NSEOptionsProvider(timeout_seconds=5.0, http=client)
        chain = await provider.get_chain(
            Instrument(symbol="RELIANCE", market_code=IN_NSE.code), date(2026, 8, 27)
        )

    assert chain is not None
    assert chain.spot == 1300.0


async def test_get_chain_routes_index_symbol_to_indices_endpoint() -> None:
    handler = _handler_for("NIFTY", "/api/option-chain-indices")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = NSEOptionsProvider(timeout_seconds=5.0, http=client)
        chain = await provider.get_chain(
            Instrument(symbol="NIFTY", market_code=IN_NSE.code), date(2026, 8, 27)
        )

    assert chain is not None


async def test_get_chain_normalizes_percent_iv_to_fraction() -> None:
    handler = _handler_for("RELIANCE", "/api/option-chain-equities")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = NSEOptionsProvider(timeout_seconds=5.0, http=client)
        chain = await provider.get_chain(
            Instrument(symbol="RELIANCE", market_code=IN_NSE.code), date(2026, 8, 27)
        )

    assert chain is not None
    calls = {q.strike: q for q in chain.calls(date(2026, 8, 27))}
    assert calls[1280.0].iv == pytest.approx(0.225)
    assert calls[1280.0].bid == 34.5
    assert calls[1280.0].ask == 35.5
    assert calls[1280.0].oi == 1000
    assert calls[1280.0].volume == 200


async def test_get_chain_zero_iv_leg_has_no_iv_or_greeks() -> None:
    handler = _handler_for("RELIANCE", "/api/option-chain-equities")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = NSEOptionsProvider(timeout_seconds=5.0, http=client)
        chain = await provider.get_chain(
            Instrument(symbol="RELIANCE", market_code=IN_NSE.code), date(2026, 9, 24)
        )

    assert chain is not None
    calls = {q.strike: q for q in chain.calls(date(2026, 9, 24))}
    assert calls[1300.0].iv is None
    assert calls[1300.0].delta is None


async def test_get_chain_fills_greeks_when_iv_present() -> None:
    handler = _handler_for("RELIANCE", "/api/option-chain-equities")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = NSEOptionsProvider(timeout_seconds=5.0, http=client)
        chain = await provider.get_chain(
            Instrument(symbol="RELIANCE", market_code=IN_NSE.code), date(2026, 8, 27)
        )

    assert chain is not None
    calls = {q.strike: q for q in chain.calls(date(2026, 8, 27))}
    assert calls[1280.0].delta is not None
    assert calls[1280.0].gamma is not None


async def test_get_chain_none_expiry_picks_nearest() -> None:
    handler = _handler_for("RELIANCE", "/api/option-chain-equities")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = NSEOptionsProvider(timeout_seconds=5.0, http=client)
        chain = await provider.get_chain(Instrument(symbol="RELIANCE", market_code=IN_NSE.code))

    assert chain is not None
    assert chain.quotes  # some expiry got selected and populated


async def test_list_expiries_parses_and_sorts_nse_date_format() -> None:
    handler = _handler_for("RELIANCE", "/api/option-chain-equities")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = NSEOptionsProvider(timeout_seconds=5.0, http=client)
        expiries = await provider.list_expiries(
            Instrument(symbol="RELIANCE", market_code=IN_NSE.code)
        )

    assert expiries == [date(2026, 8, 27), date(2026, 9, 24)]


async def test_warms_up_before_fetching_chain() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/":
            return httpx.Response(200, text="<html></html>")
        return httpx.Response(200, json=_payload())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = NSEOptionsProvider(timeout_seconds=5.0, http=client)
        await provider.get_chain(Instrument(symbol="RELIANCE", market_code=IN_NSE.code))

    assert calls == ["/", "/api/option-chain-equities"]


async def test_get_chain_retries_once_after_403_then_succeeds() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        if request.url.path == "/":
            return httpx.Response(200, text="ok")
        attempts += 1
        if attempts == 1:
            return httpx.Response(403)
        return httpx.Response(200, json=_payload())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = NSEOptionsProvider(timeout_seconds=5.0, http=client)
        chain = await provider.get_chain(Instrument(symbol="RELIANCE", market_code=IN_NSE.code))

    assert chain is not None
    assert attempts == 2


async def test_get_chain_persistent_403_returns_none_never_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/":
            return httpx.Response(200, text="ok")
        return httpx.Response(403)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = NSEOptionsProvider(timeout_seconds=5.0, http=client)
        chain = await provider.get_chain(Instrument(symbol="RELIANCE", market_code=IN_NSE.code))

    assert chain is None


async def test_get_chain_missing_underlying_value_returns_none() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/":
            return httpx.Response(200, text="ok")
        return httpx.Response(200, json={"records": {"data": []}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = NSEOptionsProvider(timeout_seconds=5.0, http=client)
        chain = await provider.get_chain(Instrument(symbol="RELIANCE", market_code=IN_NSE.code))

    assert chain is None


async def test_health_check_ok_when_underlying_value_present() -> None:
    handler = _handler_for("NIFTY", "/api/option-chain-indices")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = NSEOptionsProvider(timeout_seconds=5.0, http=client)
        health = await provider.health_check()

    assert health.ok is True
    assert "1300.0" in health.detail


async def test_health_check_request_failure_is_unhealthy_never_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = NSEOptionsProvider(timeout_seconds=5.0, http=client)
        health = await provider.health_check()

    assert health.ok is False


async def test_aclose_is_noop_for_injected_client() -> None:
    transport = httpx.MockTransport(lambda r: httpx.Response(200))
    async with httpx.AsyncClient(transport=transport) as client:
        provider = NSEOptionsProvider(timeout_seconds=5.0, http=client)
        await provider.aclose()
        assert client.is_closed is False


@pytest.mark.network
async def test_nse_options_live_health_check() -> None:
    provider = NSEOptionsProvider()
    try:
        health = await provider.health_check()
        assert health.ok is True
    finally:
        await provider.aclose()
