"""NSE cookie warm-up + quote/health-check parsing against ``httpx.MockTransport``
fixtures. Every scenario here is offline; the one live check is
network-marked and skipped by default (see pytest addopts)."""

from __future__ import annotations

from datetime import date

import httpx
import pytest

from argus.data.prices.nse_provider import NSEProvider
from argus.markets import IN_NSE, US_NASDAQ, Instrument


def test_nse_supports_only_in_nse() -> None:
    provider = NSEProvider()
    assert provider.supports(IN_NSE) is True
    assert provider.supports(US_NASDAQ) is False


async def test_nse_get_quote_wrong_market_returns_none_without_http_call() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("no HTTP call expected for a non-IN_NSE instrument")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = NSEProvider(timeout_seconds=5.0, http=client)
        quote = await provider.get_quote(Instrument(symbol="AAPL", market_code=US_NASDAQ.code))

    assert quote is None


async def test_nse_get_quote_warms_up_then_fetches() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/":
            return httpx.Response(200, text="<html></html>")
        if request.url.path == "/api/quote-equity":
            assert request.url.params.get("symbol") == "RELIANCE"
            return httpx.Response(
                200, json={"priceInfo": {"lastPrice": 1300.5, "previousClose": 1290.0}}
            )
        return httpx.Response(404)  # pragma: no cover

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = NSEProvider(timeout_seconds=5.0, http=client)
        inst = Instrument(symbol="RELIANCE", market_code=IN_NSE.code)

        quote = await provider.get_quote(inst)

    assert quote is not None
    assert quote.symbol == "RELIANCE"
    assert quote.price == 1300.5
    assert quote.prev_close == 1290.0
    # The warm-up GET (cookie dance) happens before the quote GET.
    assert calls == ["/", "/api/quote-equity"]


async def test_nse_get_quote_reuses_warm_cookies_across_calls() -> None:
    """A second ``get_quote`` on the same provider should not warm up again."""
    warmup_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal warmup_calls
        if request.url.path == "/":
            warmup_calls += 1
            return httpx.Response(200, text="ok")
        return httpx.Response(
            200, json={"priceInfo": {"lastPrice": 500.0, "previousClose": 495.0}}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = NSEProvider(timeout_seconds=5.0, http=client)
        inst = Instrument(symbol="TCS", market_code=IN_NSE.code)

        await provider.get_quote(inst)
        await provider.get_quote(inst)

    assert warmup_calls == 1


async def test_nse_get_quote_retries_once_after_403_then_succeeds() -> None:
    quote_attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal quote_attempts
        if request.url.path == "/":
            return httpx.Response(200, text="ok")
        quote_attempts += 1
        if quote_attempts == 1:
            return httpx.Response(403)  # stale/rejected cookie
        return httpx.Response(200, json={"priceInfo": {"lastPrice": 100.0, "previousClose": 99.0}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = NSEProvider(timeout_seconds=5.0, http=client)
        quote = await provider.get_quote(Instrument(symbol="INFY", market_code=IN_NSE.code))

    assert quote is not None
    assert quote.price == 100.0
    assert quote_attempts == 2


async def test_nse_get_quote_persistent_403_returns_none_never_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/":
            return httpx.Response(200, text="ok")
        return httpx.Response(403)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = NSEProvider(timeout_seconds=5.0, http=client)
        quote = await provider.get_quote(Instrument(symbol="INFY", market_code=IN_NSE.code))

    assert quote is None


async def test_nse_get_quote_missing_price_info_returns_none() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/":
            return httpx.Response(200, text="ok")
        return httpx.Response(200, json={})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = NSEProvider(timeout_seconds=5.0, http=client)
        quote = await provider.get_quote(Instrument(symbol="INFY", market_code=IN_NSE.code))

    assert quote is None


async def test_nse_get_daily_bars_always_empty() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("get_daily_bars must never make an HTTP call")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = NSEProvider(timeout_seconds=5.0, http=client)
        bars = await provider.get_daily_bars(
            Instrument(symbol="TCS", market_code=IN_NSE.code), date(2026, 1, 1), date(2026, 1, 5)
        )

    assert len(bars) == 0


async def test_nse_health_check_finds_nifty_50() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/":
            return httpx.Response(200, text="ok")
        if request.url.path == "/api/allIndices":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"index": "NIFTY 50", "last": 24500.25},
                        {"index": "NIFTY BANK", "last": 51000.0},
                    ]
                },
            )
        return httpx.Response(404)  # pragma: no cover

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = NSEProvider(timeout_seconds=5.0, http=client)
        health = await provider.health_check()

    assert health.ok is True
    assert "24500.25" in health.detail


async def test_nse_health_check_missing_nifty_is_unhealthy() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/":
            return httpx.Response(200, text="ok")
        return httpx.Response(200, json={"data": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = NSEProvider(timeout_seconds=5.0, http=client)
        health = await provider.health_check()

    assert health.ok is False


async def test_nse_health_check_request_failure_is_unhealthy_never_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = NSEProvider(timeout_seconds=5.0, http=client)
        health = await provider.health_check()

    assert health.ok is False


async def test_nse_aclose_is_noop_for_injected_client() -> None:
    """A client passed in via ``http=`` is owned by the caller -- ``aclose()``
    must be a no-op for it (matches the ``argus.advisor.llm`` backend pattern)."""
    transport = httpx.MockTransport(lambda r: httpx.Response(200))
    async with httpx.AsyncClient(transport=transport) as client:
        provider = NSEProvider(timeout_seconds=5.0, http=client)
        await provider.aclose()
        assert client.is_closed is False


@pytest.mark.network
async def test_nse_live_health_check_finds_nifty_50() -> None:
    provider = NSEProvider()
    try:
        health = await provider.health_check()
        assert health.ok is True
    finally:
        await provider.aclose()
