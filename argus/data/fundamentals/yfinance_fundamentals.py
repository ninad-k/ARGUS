"""Yahoo Finance fundamentals provider (via ``Ticker.info``).

``yfinance``'s ``Ticker.info`` is a synchronous, blocking property backed by
an HTTP call -- pushed onto a thread via ``asyncio.to_thread`` (same pattern
as ``YFinanceProvider.get_quote``) and bounded by
``DataSettings.provider_timeout_seconds``. Never raises: any failure is
logged and ``None`` is returned instead.

``debt_to_equity`` and ``dividendYield`` are the fields yfinance reports as
percentages rather than plain ratios/fractions -- ``debt_to_equity`` as e.g.
``148.75`` for a D/E of ~1.49, ``dividendYield`` as e.g. ``0.35`` for a
dividend yield of 0.35% (not 35%) -- both normalized here by dividing by 100
so they match every other ratio field's fraction convention.
"""

from __future__ import annotations

import asyncio
from datetime import date
from typing import Any

import structlog
import yfinance as yf

from argus.config import get_settings
from argus.data.fundamentals.base import FundamentalsView, default_get_many
from argus.data.prices.yfinance_provider import yahoo_ticker
from argus.markets import Instrument, Market

logger = structlog.get_logger(__name__)


class YFinanceFundamentalsProvider:
    """Fundamentals sourced from Yahoo Finance's ``Ticker.info``. Supports all markets."""

    name = "yfinance"

    def __init__(self, *, timeout_seconds: float | None = None) -> None:
        data_settings = get_settings().data
        self._timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else data_settings.provider_timeout_seconds
        )

    def supports(self, market: Market) -> bool:
        return True

    async def get(self, inst: Instrument) -> FundamentalsView | None:
        try:
            info = await asyncio.wait_for(
                asyncio.to_thread(self._fetch_info, inst), timeout=self._timeout_seconds
            )
        except Exception as exc:  # noqa: BLE001 -- provider methods never raise
            logger.warning(
                "fundamentals.yfinance.get.error",
                symbol=inst.symbol,
                market=inst.market_code,
                error=str(exc) or "timed out",
            )
            return None

        if not info:
            return None
        return _view_from_info(inst, info)

    def _fetch_info(self, inst: Instrument) -> dict[str, Any]:
        ticker = yf.Ticker(yahoo_ticker(inst))
        info = ticker.info
        return dict(info) if info else {}

    async def get_many(self, insts: list[Instrument]) -> dict[str, FundamentalsView]:
        return await default_get_many(self, insts)


def _view_from_info(inst: Instrument, info: dict[str, Any]) -> FundamentalsView:
    debt_to_equity_pct = _to_float(info.get("debtToEquity"))
    dividend_yield_pct = _to_float(info.get("dividendYield"))
    return FundamentalsView(
        symbol=inst.symbol,
        market_code=inst.market_code,
        as_of=date.today(),  # noqa: DTZ011 -- snapshot date only, market tz irrelevant here
        market_cap=_to_float(info.get("marketCap")),
        pe=_to_float(info.get("trailingPE")),
        forward_pe=_to_float(info.get("forwardPE")),
        pb=_to_float(info.get("priceToBook")),
        ps=_to_float(info.get("priceToSalesTrailing12Months")),
        roe=_to_float(info.get("returnOnEquity")),
        debt_to_equity=debt_to_equity_pct / 100.0 if debt_to_equity_pct is not None else None,
        revenue_growth=_to_float(info.get("revenueGrowth")),
        earnings_growth=_to_float(info.get("earningsGrowth")),
        dividend_yield=dividend_yield_pct / 100.0 if dividend_yield_pct is not None else None,
        profit_margin=_to_float(info.get("profitMargins")),
        sector=_to_str(info.get("sector")),
    )


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
