"""TradingView-screener-backed providers: fundamentals ``get_many``/``get``,
price-provider ``get_quote``/``top_liquid``/``health_check``, and the
``TVUniverseProvider`` fallback-to-seeds behavior.

The ``tradingview-screener`` package's query execution
(``Query.get_scanner_data``) is monkeypatched at the class level so every
provider method under test exercises its real query-building logic without
ever hitting the network. One live smoke test per provider is network-marked
and skipped by default (see pytest addopts).
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from typing import Any

import pandas as pd
import pytest
from tradingview_screener import Query

from argus.data.fundamentals.tv_fundamentals import TVScreenerFundamentalsProvider
from argus.data.prices.tv_screener_provider import TVScreenerProvider
from argus.data.universe import SeedUniverseProvider, TVUniverseProvider
from argus.markets import IN_NSE, US_NASDAQ, Instrument, Market


def _mock_scanner_data(
    rows: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Patch ``Query.get_scanner_data`` (class-wide) to return ``rows`` as a
    canned ``DataFrame``, regardless of how the query was built."""
    df = pd.DataFrame(rows)

    def _fake_get_scanner_data(self: Query, **kwargs: Any) -> tuple[int, pd.DataFrame]:
        return len(rows), df

    monkeypatch.setattr(Query, "get_scanner_data", _fake_get_scanner_data)


def _mock_scanner_data_raises(monkeypatch: pytest.MonkeyPatch, exc: Exception) -> None:
    def _fake_get_scanner_data(self: Query, **kwargs: Any) -> tuple[int, pd.DataFrame]:
        raise exc

    monkeypatch.setattr(Query, "get_scanner_data", _fake_get_scanner_data)


_AAPL_FUNDAMENTALS_ROW: dict[str, Any] = {
    "name": "AAPL",
    "sector": "Electronic Technology",
    "market_cap_basic": 3.0e12,
    "price_earnings_ttm": 32.1,
    "price_book_fq": 45.2,
    "price_sales_ratio": 9.1,
    "return_on_equity": 147.5,
    "debt_to_equity": 1.49,
    "total_revenue_yoy_growth_ttm": 8.0,
    "earnings_per_share_diluted_yoy_growth_ttm": 10.2,
    "dividends_yield_current": 0.44,
    "net_margin": 24.3,
}


# --- TVScreenerFundamentalsProvider ------------------------------------------


async def test_tv_fundamentals_get_many_parses_canned_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_scanner_data([_AAPL_FUNDAMENTALS_ROW], monkeypatch)
    provider = TVScreenerFundamentalsProvider(timeout_seconds=5.0)

    result = await provider.get_many([Instrument(symbol="AAPL", market_code=US_NASDAQ.code)])

    assert set(result) == {"AAPL"}
    view = result["AAPL"]
    assert view.market_code == US_NASDAQ.code
    assert view.market_cap == 3.0e12
    assert view.pe == 32.1
    assert view.pb == 45.2
    assert view.ps == 9.1
    assert view.roe == pytest.approx(1.475)
    assert view.debt_to_equity == 1.49
    assert view.revenue_growth == pytest.approx(0.08)
    assert view.earnings_growth == pytest.approx(0.102)
    assert view.dividend_yield == pytest.approx(0.0044)
    assert view.profit_margin == pytest.approx(0.243)
    assert view.sector == "Electronic Technology"
    assert view.forward_pe is None  # no forward-PE-equivalent field, see module docstring


async def test_tv_fundamentals_get_delegates_to_get_many(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_scanner_data([_AAPL_FUNDAMENTALS_ROW], monkeypatch)
    provider = TVScreenerFundamentalsProvider(timeout_seconds=5.0)

    view = await provider.get(Instrument(symbol="AAPL", market_code=US_NASDAQ.code))

    assert view is not None
    assert view.symbol == "AAPL"


async def test_tv_fundamentals_get_many_empty_insts_returns_empty_without_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_scanner_data_raises(monkeypatch, AssertionError("should not be called"))
    provider = TVScreenerFundamentalsProvider(timeout_seconds=5.0)

    assert await provider.get_many([]) == {}


async def test_tv_fundamentals_get_many_query_failure_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_scanner_data_raises(monkeypatch, RuntimeError("simulated network failure"))
    provider = TVScreenerFundamentalsProvider(timeout_seconds=5.0)

    result = await provider.get_many([Instrument(symbol="AAPL", market_code=US_NASDAQ.code)])

    assert result == {}


async def test_tv_fundamentals_skips_unsupported_market() -> None:
    provider = TVScreenerFundamentalsProvider(timeout_seconds=5.0)
    result = await provider.get_many([Instrument(symbol="X", market_code="UNKNOWN_MARKET")])
    assert result == {}


def test_tv_fundamentals_supports_known_markets_only() -> None:
    provider = TVScreenerFundamentalsProvider(timeout_seconds=5.0)
    assert provider.supports(US_NASDAQ) is True
    assert provider.supports(IN_NSE) is True
    assert provider.supports(replace(US_NASDAQ, code="MADE_UP")) is False


# --- TVScreenerProvider: get_quote -------------------------------------------


async def test_tv_provider_get_quote_parses_canned_row(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_scanner_data([{"name": "AAPL", "close": 209.66, "change": -1.591176}], monkeypatch)
    provider = TVScreenerProvider(timeout_seconds=5.0)

    quote = await provider.get_quote(Instrument(symbol="AAPL", market_code=US_NASDAQ.code))

    assert quote is not None
    assert quote.symbol == "AAPL"
    assert quote.price == 209.66
    assert quote.prev_close is not None
    assert quote.prev_close == pytest.approx(209.66 / (1 - 0.01591176))


async def test_tv_provider_get_quote_empty_result_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_scanner_data([], monkeypatch)
    provider = TVScreenerProvider(timeout_seconds=5.0)

    quote = await provider.get_quote(Instrument(symbol="NOPE", market_code=US_NASDAQ.code))

    assert quote is None


async def test_tv_provider_get_quote_query_failure_returns_none_never_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_scanner_data_raises(monkeypatch, RuntimeError("boom"))
    provider = TVScreenerProvider(timeout_seconds=5.0)

    quote = await provider.get_quote(Instrument(symbol="AAPL", market_code=US_NASDAQ.code))

    assert quote is None


async def test_tv_provider_get_quote_unsupported_market_returns_none() -> None:
    provider = TVScreenerProvider(timeout_seconds=5.0)
    quote = await provider.get_quote(Instrument(symbol="X", market_code="UNKNOWN_MARKET"))
    assert quote is None


async def test_tv_provider_get_daily_bars_always_empty() -> None:
    provider = TVScreenerProvider(timeout_seconds=5.0)
    bars = await provider.get_daily_bars(
        Instrument(symbol="AAPL", market_code=US_NASDAQ.code), date(2026, 1, 1), date(2026, 1, 5)
    )
    assert len(bars) == 0


# --- TVScreenerProvider: top_liquid -------------------------------------------


async def test_tv_provider_top_liquid_parses_canned_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        {"name": "NVDA", "description": "NVIDIA Corporation", "sector": "Electronic Technology"},
        {"name": "META", "description": "Meta Platforms, Inc.", "sector": "Technology Services"},
    ]
    _mock_scanner_data(rows, monkeypatch)
    provider = TVScreenerProvider(timeout_seconds=5.0)

    instruments = await provider.top_liquid(US_NASDAQ, n=2)

    assert [i.symbol for i in instruments] == ["NVDA", "META"]
    assert instruments[0].name == "NVIDIA Corporation"
    assert instruments[0].sector == "Electronic Technology"
    assert all(i.market_code == US_NASDAQ.code for i in instruments)


async def test_tv_provider_top_liquid_query_failure_returns_empty_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_scanner_data_raises(monkeypatch, RuntimeError("boom"))
    provider = TVScreenerProvider(timeout_seconds=5.0)

    instruments = await provider.top_liquid(US_NASDAQ, n=10)

    assert instruments == []


async def test_tv_provider_top_liquid_unsupported_market_returns_empty_list() -> None:
    provider = TVScreenerProvider(timeout_seconds=5.0)
    market = replace(US_NASDAQ, code="MADE_UP")
    assert await provider.top_liquid(market, n=10) == []


# --- TVScreenerProvider: health_check -----------------------------------------


async def test_tv_provider_health_check_ok_when_rows_returned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = {"name": "AAPL", "description": "Apple Inc.", "sector": "Tech"}
    _mock_scanner_data([row], monkeypatch)
    provider = TVScreenerProvider(timeout_seconds=5.0)

    health = await provider.health_check()

    assert health.ok is True


async def test_tv_provider_health_check_fails_never_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_scanner_data_raises(monkeypatch, RuntimeError("boom"))
    provider = TVScreenerProvider(timeout_seconds=5.0)

    health = await provider.health_check()

    assert health.ok is False


# --- TVUniverseProvider: fallback to seeds -----------------------------------


class _FakeUniverseSource:
    def __init__(
        self, instruments: list[Instrument] | None = None, raise_exc: Exception | None = None
    ) -> None:
        self._instruments = instruments or []
        self._raise = raise_exc

    async def top_liquid(self, market: Market, n: int) -> list[Instrument]:
        if self._raise is not None:
            raise self._raise
        return list(self._instruments)


async def test_tv_universe_provider_uses_top_liquid_result_when_nonempty() -> None:
    inst = Instrument(symbol="AAPL", market_code=US_NASDAQ.code)
    provider = TVUniverseProvider(
        _FakeUniverseSource(instruments=[inst]), SeedUniverseProvider(), size=10
    )

    result = await provider.universe(US_NASDAQ)

    assert result == [inst]


async def test_tv_universe_provider_falls_back_to_seed_on_empty_result() -> None:
    provider = TVUniverseProvider(
        _FakeUniverseSource(instruments=[]), SeedUniverseProvider(), size=10
    )

    result = await provider.universe(US_NASDAQ)

    assert len(result) > 0
    assert all(inst.market_code == US_NASDAQ.code for inst in result)


async def test_tv_universe_provider_falls_back_to_seed_on_exception() -> None:
    provider = TVUniverseProvider(
        _FakeUniverseSource(raise_exc=RuntimeError("scanner unavailable")),
        SeedUniverseProvider(),
        size=10,
    )

    result = await provider.universe(US_NASDAQ)

    assert len(result) > 0
    assert all(inst.market_code == US_NASDAQ.code for inst in result)


# --- Live smoke tests (network-marked, skipped by default) ------------------


@pytest.mark.network
async def test_tv_provider_live_get_quote_aapl() -> None:
    provider = TVScreenerProvider()
    quote = await provider.get_quote(Instrument(symbol="AAPL", market_code=US_NASDAQ.code))
    assert quote is not None
    assert quote.price > 0


@pytest.mark.network
async def test_tv_fundamentals_live_get_aapl() -> None:
    provider = TVScreenerFundamentalsProvider()
    view = await provider.get(Instrument(symbol="AAPL", market_code=US_NASDAQ.code))
    assert view is not None
    assert view.market_cap is not None and view.market_cap > 0
