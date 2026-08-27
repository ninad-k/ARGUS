"""TradingView-screener-backed bulk fundamentals provider.

Uses the ``tradingview-screener`` package (installed version: 3.2.1, no
fields manifest shipped in this version) to fetch fundamentals for many
symbols in a single scanner query per market -- far cheaper than yfinance's
one-symbol-at-a-time ``Ticker.info``. ``get_many`` is the primary path;
``get`` delegates to it for a single instrument.

Column names below were verified empirically against the installed package
(live scanner queries against real large-cap symbols in both the ``america``
and ``india`` markets -- an unknown/invalid column silently returns ``None``
for every row rather than raising, so "does it error" can't be used to
validate a name):

  ``market_cap_basic``, ``price_earnings_ttm``, ``price_book_fq``,
  ``price_sales_ratio``, ``return_on_equity``, ``debt_to_equity``,
  ``total_revenue_yoy_growth_ttm``,
  ``earnings_per_share_diluted_yoy_growth_ttm``, ``dividends_yield_current``,
  ``net_margin``, ``sector``, ``name`` (ticker), ``description`` (company name).

Percent-valued columns (``return_on_equity``, ``total_revenue_yoy_growth_ttm``,
``earnings_per_share_diluted_yoy_growth_ttm``, ``dividends_yield_current``,
``net_margin``) come back as e.g. ``62.9`` for 62.9% -- divided by 100 here to
match ``FundamentalsView``'s fraction convention. ``debt_to_equity`` and the
price multiples (P/E, P/B, P/S) come back as plain ratios already, so those
are passed through unchanged.

No forward-PE-equivalent field exists in this version of the package -- every
candidate name tried (``forward_pe``, ``price_earnings_forecast_fq``,
``price_earnings_forecast_next_fy``, ``price_2_earnings_next_fy``, etc.)
returned ``None`` for real symbols with known non-null forward PEs, so
``forward_pe`` is always ``None`` here rather than silently wrong.
"""

from __future__ import annotations

import asyncio
from datetime import date
from typing import Any, cast

import structlog
from tradingview_screener import Column, Query

from argus.config import get_settings
from argus.data.fundamentals.base import FundamentalsView
from argus.markets import IN_NSE, Instrument, Market
from argus.markets.registry import US_NASDAQ, US_NYSE

logger = structlog.get_logger(__name__)

_TV_MARKET_BY_CODE = {
    US_NYSE.code: "america",
    US_NASDAQ.code: "america",
    IN_NSE.code: "india",
}

_SELECT_COLUMNS = (
    "name",
    "sector",
    "market_cap_basic",
    "price_earnings_ttm",
    "price_book_fq",
    "price_sales_ratio",
    "return_on_equity",
    "debt_to_equity",
    "total_revenue_yoy_growth_ttm",
    "earnings_per_share_diluted_yoy_growth_ttm",
    "dividends_yield_current",
    "net_margin",
)


class TVScreenerFundamentalsProvider:
    """Bulk fundamentals via the TradingView scanner. Never raises; returns
    ``{}``/``None`` on any failure."""

    name = "tvscreener"

    def __init__(self, *, timeout_seconds: float | None = None) -> None:
        data_settings = get_settings().data
        self._timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else data_settings.provider_timeout_seconds
        )

    def supports(self, market: Market) -> bool:
        return market.code in _TV_MARKET_BY_CODE

    async def get(self, inst: Instrument) -> FundamentalsView | None:
        result = await self.get_many([inst])
        return result.get(inst.symbol)

    async def get_many(self, insts: list[Instrument]) -> dict[str, FundamentalsView]:
        if not insts:
            return {}

        by_market: dict[str, list[Instrument]] = {}
        for inst in insts:
            by_market.setdefault(inst.market_code, []).append(inst)

        results: dict[str, FundamentalsView] = {}
        for market_code, market_insts in by_market.items():
            tv_market = _TV_MARKET_BY_CODE.get(market_code)
            if tv_market is None:
                continue
            results.update(await self._fetch_market(tv_market, market_code, market_insts))
        return results

    async def _fetch_market(
        self, tv_market: str, market_code: str, insts: list[Instrument]
    ) -> dict[str, FundamentalsView]:
        symbols = [inst.symbol for inst in insts]
        try:
            rows = await asyncio.wait_for(
                asyncio.to_thread(self._query_sync, tv_market, symbols),
                timeout=self._timeout_seconds,
            )
        except Exception as exc:  # noqa: BLE001 -- provider methods never raise
            logger.warning(
                "fundamentals.tvscreener.get_many.error",
                market=market_code,
                symbols=len(symbols),
                error=str(exc) or "timed out",
            )
            return {}

        as_of = date.today()  # noqa: DTZ011 -- snapshot date only, market tz irrelevant here
        views: dict[str, FundamentalsView] = {}
        for row in rows:
            symbol = _to_str(row.get("name"))
            if not symbol:
                continue
            views[symbol] = _view_from_row(symbol, market_code, row, as_of)
        return views

    def _query_sync(self, tv_market: str, symbols: list[str]) -> list[dict[str, Any]]:
        query = (
            Query()
            .select(*_SELECT_COLUMNS)
            .where(Column("name").isin(symbols))
            .set_markets(tv_market)
            .limit(len(symbols))
        )
        _count, df = query.get_scanner_data()
        rows: list[dict[str, Any]] = cast("list[dict[str, Any]]", df.to_dict("records"))
        return rows


def _view_from_row(
    symbol: str, market_code: str, row: dict[str, Any], as_of: date
) -> FundamentalsView:
    return FundamentalsView(
        symbol=symbol,
        market_code=market_code,
        as_of=as_of,
        market_cap=_to_float(row.get("market_cap_basic")),
        pe=_to_float(row.get("price_earnings_ttm")),
        forward_pe=None,  # no forward-PE-equivalent field in this package version
        pb=_to_float(row.get("price_book_fq")),
        ps=_to_float(row.get("price_sales_ratio")),
        roe=_pct_to_fraction(row.get("return_on_equity")),
        debt_to_equity=_to_float(row.get("debt_to_equity")),
        revenue_growth=_pct_to_fraction(row.get("total_revenue_yoy_growth_ttm")),
        earnings_growth=_pct_to_fraction(row.get("earnings_per_share_diluted_yoy_growth_ttm")),
        dividend_yield=_pct_to_fraction(row.get("dividends_yield_current")),
        profit_margin=_pct_to_fraction(row.get("net_margin")),
        sector=_to_str(row.get("sector")),
    )


def _pct_to_fraction(value: Any) -> float | None:
    f = _to_float(value)
    return f / 100.0 if f is not None else None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN guard -- pandas hands back NaN (not None) for missing numeric cells
        return None
    return f


def _to_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and value != value:  # NaN guard, see _to_float
        return None
    s = str(value)
    return s or None
