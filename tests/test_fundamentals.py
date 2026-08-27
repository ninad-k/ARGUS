"""FundamentalsView mapping from a canned yfinance info dict, the
Null/Static providers, ``default_get_many``, and the kind->provider factory.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from argus.data.fundamentals.base import (
    FundamentalsView,
    NullFundamentalsProvider,
    StaticFundamentalsProvider,
    default_get_many,
)
from argus.data.fundamentals.factory import (
    build_default_fundamentals,
    build_fundamentals_provider,
)
from argus.data.fundamentals.tv_fundamentals import TVScreenerFundamentalsProvider
from argus.data.fundamentals.yfinance_fundamentals import (
    YFinanceFundamentalsProvider,
    _view_from_info,
)
from argus.markets import US_NASDAQ, Instrument

_CANNED_INFO: dict[str, Any] = {
    "marketCap": 3_000_000_000_000.0,
    "trailingPE": 32.1,
    "forwardPE": 28.4,
    "priceToBook": 45.2,
    "priceToSalesTrailing12Months": 15.3,
    "returnOnEquity": 1.47,
    "debtToEquity": 148.75,  # yfinance reports this as a percent, not a fraction
    "revenueGrowth": 0.153,
    "earningsGrowth": 0.12,
    "dividendYield": 0.44,  # yfinance reports this as a percent too, not a fraction
    "profitMargins": 0.243,
    "sector": "Technology",
}


# --- _view_from_info -----------------------------------------------------------


def test_view_from_info_maps_fields_and_normalizes_debt_to_equity() -> None:
    inst = Instrument(symbol="AAPL", market_code=US_NASDAQ.code)
    view = _view_from_info(inst, _CANNED_INFO)

    assert view.symbol == "AAPL"
    assert view.market_code == US_NASDAQ.code
    assert view.market_cap == 3_000_000_000_000.0
    assert view.pe == 32.1
    assert view.forward_pe == 28.4
    assert view.pb == 45.2
    assert view.ps == 15.3
    assert view.roe == 1.47
    assert view.debt_to_equity == pytest.approx(1.4875)  # 148.75% -> 1.4875
    assert view.revenue_growth == 0.153
    assert view.earnings_growth == 0.12
    assert view.dividend_yield == pytest.approx(0.0044)  # 0.44% -> 0.0044
    assert view.profit_margin == 0.243
    assert view.sector == "Technology"
    assert view.as_of == date.today()  # noqa: DTZ011


def test_view_from_info_missing_fields_are_none() -> None:
    inst = Instrument(symbol="X", market_code=US_NASDAQ.code)
    view = _view_from_info(inst, {})

    assert view.market_cap is None
    assert view.pe is None
    assert view.debt_to_equity is None
    assert view.sector is None


# --- YFinanceFundamentalsProvider ------------------------------------------------


async def test_yfinance_fundamentals_get_uses_fetch_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = YFinanceFundamentalsProvider(timeout_seconds=5.0)
    inst = Instrument(symbol="AAPL", market_code=US_NASDAQ.code)
    monkeypatch.setattr(provider, "_fetch_info", lambda i: dict(_CANNED_INFO))

    view = await provider.get(inst)

    assert view is not None
    assert view.symbol == "AAPL"
    assert view.pe == 32.1


async def test_yfinance_fundamentals_get_empty_info_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = YFinanceFundamentalsProvider(timeout_seconds=5.0)
    inst = Instrument(symbol="NOPE", market_code=US_NASDAQ.code)
    monkeypatch.setattr(provider, "_fetch_info", lambda i: {})

    assert await provider.get(inst) is None


async def test_yfinance_fundamentals_get_error_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = YFinanceFundamentalsProvider(timeout_seconds=5.0)
    inst = Instrument(symbol="BOOM", market_code=US_NASDAQ.code)

    def _raise(_inst: Instrument) -> dict[str, Any]:
        raise RuntimeError("simulated network failure")

    monkeypatch.setattr(provider, "_fetch_info", _raise)

    assert await provider.get(inst) is None


async def test_yfinance_fundamentals_get_many_uses_default_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = YFinanceFundamentalsProvider(timeout_seconds=5.0)
    monkeypatch.setattr(provider, "_fetch_info", lambda i: dict(_CANNED_INFO))
    instruments = [
        Instrument(symbol="AAPL", market_code=US_NASDAQ.code),
        Instrument(symbol="MSFT", market_code=US_NASDAQ.code),
    ]

    result = await provider.get_many(instruments)

    assert set(result) == {"AAPL", "MSFT"}
    assert result["AAPL"].pe == 32.1


def test_yfinance_fundamentals_supports_all_markets() -> None:
    provider = YFinanceFundamentalsProvider(timeout_seconds=5.0)
    assert provider.supports(US_NASDAQ) is True


# --- NullFundamentalsProvider / StaticFundamentalsProvider ------------------


async def test_null_fundamentals_provider_always_returns_none() -> None:
    provider = NullFundamentalsProvider()
    inst = Instrument(symbol="AAPL", market_code=US_NASDAQ.code)

    assert await provider.get(inst) is None
    assert await provider.get_many([inst]) == {}
    assert provider.supports(US_NASDAQ) is True


async def test_static_fundamentals_provider_returns_added_view() -> None:
    provider = StaticFundamentalsProvider()
    view = FundamentalsView(symbol="AAPL", market_code=US_NASDAQ.code, as_of=date(2026, 1, 2))
    provider.add(view)

    result = await provider.get(Instrument(symbol="AAPL", market_code=US_NASDAQ.code))
    assert result is view

    missing = await provider.get(Instrument(symbol="NOPE", market_code=US_NASDAQ.code))
    assert missing is None


async def test_static_fundamentals_provider_get_many() -> None:
    view_a = FundamentalsView(symbol="AAPL", market_code=US_NASDAQ.code, as_of=date(2026, 1, 2))
    view_b = FundamentalsView(symbol="MSFT", market_code=US_NASDAQ.code, as_of=date(2026, 1, 2))
    provider = StaticFundamentalsProvider({"AAPL": view_a, "MSFT": view_b})

    result = await provider.get_many(
        [
            Instrument(symbol="AAPL", market_code=US_NASDAQ.code),
            Instrument(symbol="MSFT", market_code=US_NASDAQ.code),
            Instrument(symbol="NOPE", market_code=US_NASDAQ.code),
        ]
    )

    assert result == {"AAPL": view_a, "MSFT": view_b}


async def test_default_get_many_skips_none_results() -> None:
    class _HalfProvider:
        name = "half"

        def supports(self, market: object) -> bool:
            return True

        async def get(self, inst: Instrument) -> FundamentalsView | None:
            if inst.symbol != "GOOD":
                return None
            return FundamentalsView(
                symbol="GOOD", market_code=inst.market_code, as_of=date(2026, 1, 2)
            )

    result = await default_get_many(
        _HalfProvider(),  # type: ignore[arg-type]
        [
            Instrument(symbol="GOOD", market_code=US_NASDAQ.code),
            Instrument(symbol="BAD", market_code=US_NASDAQ.code),
        ],
    )

    assert set(result) == {"GOOD"}


# --- factory -----------------------------------------------------------------


def test_factory_builds_yfinance() -> None:
    assert isinstance(build_fundamentals_provider("yfinance", {}), YFinanceFundamentalsProvider)


def test_factory_builds_tvscreener() -> None:
    provider = build_fundamentals_provider("tvscreener", {})
    assert isinstance(provider, TVScreenerFundamentalsProvider)


def test_factory_builds_static() -> None:
    assert isinstance(build_fundamentals_provider("static", {}), StaticFundamentalsProvider)


def test_factory_builds_null() -> None:
    assert isinstance(build_fundamentals_provider("null", {}), NullFundamentalsProvider)


def test_factory_unknown_kind_falls_back_to_null() -> None:
    provider = build_fundamentals_provider("carrier_pigeon", {})
    assert isinstance(provider, NullFundamentalsProvider)


def test_build_default_fundamentals_is_yfinance() -> None:
    assert isinstance(build_default_fundamentals(), YFinanceFundamentalsProvider)
