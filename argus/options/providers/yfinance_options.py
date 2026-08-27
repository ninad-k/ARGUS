"""Yahoo Finance option-chain provider (via the ``yfinance`` package).

``yfinance`` is synchronous and blocking -- every call here runs on a thread
via ``asyncio.to_thread`` (same pattern as ``argus.data.prices.yfinance_provider``),
bounded by ``DataSettings.provider_timeout_seconds``. Never raises.

``Ticker.options`` returns the available expiry strings; ``Ticker.option_chain(expiry)``
returns ``calls``/``puts`` DataFrames with ``strike``, ``bid``, ``ask``,
``lastPrice``, ``impliedVolatility``, ``openInterest``, ``volume`` columns
for that one expiry -- unlike NSE's option-chain endpoint, yfinance has no
"all expiries in one call" response, so each ``get_chain`` fetches exactly
one expiry (the requested one, or the nearest, when ``expiry=None``) while
``expiries`` on the returned chain still lists every expiry available so
callers can page through them.

US markets only -- yfinance's India option-chain data (via the ``.NS``
ticker suffix) is largely empty/unreliable in practice, so ``supports``
rejects IN_NSE and callers should route India through
``argus.options.providers.nse_options.NSEOptionsProvider`` instead (see
``argus.options.providers.factory``).

Greeks are computed locally (not supplied by yfinance) via
``argus.options.black_scholes.greeks``, using each contract's own
``impliedVolatility`` when present and positive; contracts with no usable IV
get ``None`` greeks rather than a fabricated value.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime
from typing import Any

import structlog
import yfinance as yf

from argus.config import get_settings
from argus.data.prices.base import ProviderHealth
from argus.data.prices.yfinance_provider import yahoo_ticker
from argus.markets import IN_NSE, Instrument
from argus.options import black_scholes
from argus.options.models import OptionChain, OptionQuote, Right
from argus.options.providers.base import nearest_expiry

logger = structlog.get_logger(__name__)


class YFinanceOptionsProvider:
    """Option chains sourced from Yahoo Finance. US markets only. Never raises."""

    name = "yfinance"

    def __init__(self, *, timeout_seconds: float | None = None) -> None:
        settings = get_settings()
        self._timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else settings.data.provider_timeout_seconds
        )
        self._risk_free_rate = settings.options.risk_free_rate

    def supports(self, inst: Instrument) -> bool:
        return inst.market_code != IN_NSE.code

    async def aclose(self) -> None:
        return None

    async def list_expiries(self, inst: Instrument) -> list[date]:
        if not self.supports(inst):
            return []
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._fetch_expiries, inst), timeout=self._timeout_seconds
            )
        except Exception as exc:  # noqa: BLE001 -- provider methods never raise
            logger.warning(
                "options.yfinance.list_expiries.error",
                symbol=inst.symbol,
                error=str(exc) or "timed out",
            )
            return []

    def _fetch_expiries(self, inst: Instrument) -> list[date]:
        ticker = yf.Ticker(yahoo_ticker(inst))
        raw = ticker.options or ()
        return sorted(d for d in (_parse_yf_date(s) for s in raw) if d is not None)

    async def get_chain(self, inst: Instrument, expiry: date | None = None) -> OptionChain | None:
        if not self.supports(inst):
            return None
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._fetch_chain, inst, expiry),
                timeout=self._timeout_seconds,
            )
        except Exception as exc:  # noqa: BLE001 -- provider methods never raise
            logger.warning(
                "options.yfinance.get_chain.error",
                symbol=inst.symbol,
                expiry=str(expiry) if expiry else None,
                error=str(exc) or "timed out",
            )
            return None

    def _fetch_chain(self, inst: Instrument, expiry: date | None) -> OptionChain | None:
        ticker = yf.Ticker(yahoo_ticker(inst))
        # Read .options off this same ticker instance rather than calling
        # _fetch_expiries(inst) (which would build a second yf.Ticker and
        # double the "list expiries" network round-trip).
        raw_expiries = ticker.options or ()
        expiries = sorted(d for d in (_parse_yf_date(s) for s in raw_expiries) if d is not None)
        if not expiries:
            return None
        chosen = expiry if expiry in expiries else nearest_expiry(expiries)
        if chosen is None:
            return None

        spot = _spot_price(ticker)
        if spot is None:
            return None

        opt = ticker.option_chain(chosen.isoformat())
        today = date.today()  # noqa: DTZ011 -- expiry-day boundary only, market tz irrelevant here
        dte = max((chosen - today).days, 0)
        T = dte / 365.0 if dte > 0 else 1.0 / 365.0

        quotes = _quotes_from_frame(
            opt.calls, chosen, "C", spot, T, self._risk_free_rate
        ) + _quotes_from_frame(opt.puts, chosen, "P", spot, T, self._risk_free_rate)

        return OptionChain(
            symbol=inst.symbol,
            market_code=inst.market_code,
            spot=spot,
            as_of=datetime.now(UTC),
            expiries=expiries,
            quotes=quotes,
        )

    async def health_check(self) -> ProviderHealth:
        try:
            expiries = await asyncio.wait_for(
                asyncio.to_thread(self._health_check_sync), timeout=self._timeout_seconds
            )
        except Exception as exc:  # noqa: BLE001 -- provider methods never raise
            return ProviderHealth(
                ok=False, detail=str(exc) or "timed out", checked_at=datetime.now(UTC)
            )
        ok = len(expiries) > 0
        detail = f"AAPL has {len(expiries)} expiries" if ok else "no AAPL expiries returned"
        return ProviderHealth(ok=ok, detail=detail, checked_at=datetime.now(UTC))

    def _health_check_sync(self) -> list[date]:
        aapl = Instrument(symbol="AAPL", market_code="US_NASDAQ")
        return self._fetch_expiries(aapl)


def _parse_yf_date(s: str) -> date | None:
    try:
        return date.fromisoformat(s)
    except (TypeError, ValueError):
        return None


def _spot_price(ticker: yf.Ticker) -> float | None:
    try:
        return float(ticker.fast_info.last_price)
    except Exception:  # noqa: BLE001 -- fall back to last daily close
        logger.debug("options.yfinance.fast_info_failed", ticker=str(ticker.ticker))
    try:
        hist = ticker.history(period="5d", interval="1d", auto_adjust=False)
        if hist is None or hist.empty:
            return None
        closes = hist["Close"].dropna()
        if closes.empty:
            return None
        return float(closes.iloc[-1])
    except Exception:  # noqa: BLE001 -- provider methods never raise
        return None


def _quotes_from_frame(
    df: Any, expiry: date, right: Right, spot: float, T: float, risk_free_rate: float
) -> list[OptionQuote]:
    quotes: list[OptionQuote] = []
    if df is None or df.empty:
        return quotes
    for row in df.to_dict("records"):
        strike = _to_float(row.get("strike"))
        if strike is None:
            continue
        bid = _to_float(row.get("bid"))
        ask = _to_float(row.get("ask"))
        last = _to_float(row.get("lastPrice"))
        iv = _to_float(row.get("impliedVolatility"))
        oi = _to_float(row.get("openInterest"))
        volume = _to_float(row.get("volume"))

        delta = gamma = theta = vega = None
        if iv is not None and iv > 0:
            g = black_scholes.greeks(
                S=spot, K=strike, T=T, sigma=iv, option_type=right, r=risk_free_rate
            )
            delta, gamma, theta, vega = g.delta, g.gamma, g.theta, g.vega

        quotes.append(
            OptionQuote(
                strike=strike,
                expiry=expiry,
                right=right,
                bid=bid,
                ask=ask,
                last=last,
                iv=iv,
                oi=oi,
                volume=volume,
                delta=delta,
                gamma=gamma,
                theta=theta,
                vega=vega,
            )
        )
    return quotes


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN -- yfinance leaves bid/ask/iv NaN for illiquid contracts
        return None
    return f
