"""NSE (National Stock Exchange of India) option-chain provider.

Adapted from DRUVA's ``core/options/chain_feed.py`` (``OptionChainFeed``),
reusing the shared cookie warm-up + 403-retry helper from
``argus.data.nse_http`` -- the same dance
``argus.data.prices.nse_provider.NSEProvider`` uses for equity quotes (see
that module's docstring for why NSE's WAF needs it) -- instead of
duplicating DRUVA's independent reimplementation of it.

NSE serves index chains (NIFTY, BANKNIFTY, ...) from a different endpoint
(``/api/option-chain-indices``) than single-stock F&O names
(``/api/option-chain-equities``); ``_INDEX_SYMBOLS`` routes between them.
Both endpoints return every expiry for the underlying in one response, so
(unlike the yfinance provider) a single fetch here can serve any expiry
without a second round-trip.

NSE reports ``impliedVolatility`` as a percentage (e.g. ``18.4`` for 18.4 %)
-- normalized to a fraction here before storing on ``OptionQuote.iv`` and
before feeding it to ``black_scholes.greeks`` for locally-computed greeks.
IN_NSE only. Never raises.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime
from typing import Any

import httpx
import structlog

from argus.config import get_settings
from argus.data.nse_http import NSESession
from argus.data.prices.base import ProviderHealth
from argus.markets import IN_NSE, Instrument
from argus.options import black_scholes
from argus.options.models import OptionChain, OptionQuote, Right
from argus.options.providers.base import nearest_expiry

logger = structlog.get_logger(__name__)

_INDEX_PATH = "/api/option-chain-indices"
_EQUITY_PATH = "/api/option-chain-equities"

# Index underlyings NSE serves from the indices endpoint rather than the
# equities one -- everything else is assumed to be a single-stock F&O name.
_INDEX_SYMBOLS = frozenset({"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"})

_EXPIRY_FORMATS = ("%d-%b-%Y", "%Y-%m-%d", "%d-%m-%Y")


class NSEOptionsProvider:
    """Option chains sourced from NSE's public option-chain API. IN_NSE only.
    Never raises.

    ``http`` is exposed for tests (inject a client wired to
    ``httpx.MockTransport``); production callers can omit it and let the
    provider own a fresh ``httpx.AsyncClient`` for its lifetime.
    """

    name = "nse"

    def __init__(
        self,
        *,
        timeout_seconds: float | None = None,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        settings = get_settings()
        self._timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else settings.data.provider_timeout_seconds
        )
        self._risk_free_rate = settings.options.risk_free_rate
        self._session = NSESession(timeout_seconds=self._timeout_seconds, http=http)

    def supports(self, inst: Instrument) -> bool:
        return inst.market_code == IN_NSE.code

    async def aclose(self) -> None:
        await self._session.aclose()

    async def list_expiries(self, inst: Instrument) -> list[date]:
        payload = await self._safe_fetch(inst)
        if payload is None:
            return []
        return _expiries_from_payload(payload)

    async def get_chain(self, inst: Instrument, expiry: date | None = None) -> OptionChain | None:
        payload = await self._safe_fetch(inst)
        if payload is None:
            return None
        return _chain_from_payload(inst, payload, expiry, self._risk_free_rate)

    async def health_check(self) -> ProviderHealth:
        payload = await self._safe_fetch(Instrument(symbol="NIFTY", market_code=IN_NSE.code))
        if payload is None:
            return ProviderHealth(
                ok=False,
                detail="NIFTY option-chain request failed",
                checked_at=datetime.now(UTC),
            )
        records = payload.get("records") or {}
        spot = _to_float(records.get("underlyingValue"))
        ok = spot is not None
        detail = f"NIFTY spot={spot}" if ok else "no underlyingValue in NIFTY option-chain response"
        return ProviderHealth(ok=ok, detail=detail, checked_at=datetime.now(UTC))

    async def _safe_fetch(self, inst: Instrument) -> dict[str, Any] | None:
        if not self.supports(inst):
            return None
        path = _INDEX_PATH if inst.symbol.upper() in _INDEX_SYMBOLS else _EQUITY_PATH
        try:
            return await asyncio.wait_for(
                self._session.get_json(path, {"symbol": inst.symbol}),
                timeout=self._timeout_seconds,
            )
        except Exception as exc:  # noqa: BLE001 -- provider methods never raise
            logger.warning(
                "options.nse.request.error",
                symbol=inst.symbol,
                path=path,
                error=str(exc) or "timed out",
            )
            return None


def _parse_nse_expiry(s: str) -> date | None:
    for fmt in _EXPIRY_FORMATS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _expiries_from_payload(payload: dict[str, Any]) -> list[date]:
    records = payload.get("records") or {}
    rows = records.get("data") or []
    raw = {row.get("expiryDate") for row in rows if row.get("expiryDate")}
    return sorted(d for d in (_parse_nse_expiry(s) for s in raw) if d is not None)


def _chain_from_payload(
    inst: Instrument, payload: dict[str, Any], expiry: date | None, risk_free_rate: float
) -> OptionChain | None:
    records = payload.get("records") or {}
    spot = _to_float(records.get("underlyingValue"))
    if spot is None:
        return None

    expiries = _expiries_from_payload(payload)
    if not expiries:
        return None
    chosen = expiry if expiry in expiries else nearest_expiry(expiries)
    if chosen is None:
        return None

    today = date.today()  # noqa: DTZ011 -- expiry-day boundary only, IST irrelevant here
    dte = max((chosen - today).days, 0)
    T = dte / 365.0 if dte > 0 else 1.0 / 365.0

    rows = records.get("data") or []
    quotes: list[OptionQuote] = []
    for row in rows:
        if _parse_nse_expiry(row.get("expiryDate", "")) != chosen:
            continue
        strike = _to_float(row.get("strikePrice"))
        if strike is None:
            continue
        right_keys: tuple[tuple[Right, str], ...] = (("C", "CE"), ("P", "PE"))
        for right, key in right_keys:
            leg = row.get(key)
            if not leg:
                continue
            quotes.append(_quote_from_leg(leg, strike, chosen, right, spot, T, risk_free_rate))

    return OptionChain(
        symbol=inst.symbol,
        market_code=inst.market_code,
        spot=spot,
        as_of=datetime.now(UTC),
        expiries=expiries,
        quotes=quotes,
    )


def _quote_from_leg(
    leg: dict[str, Any],
    strike: float,
    expiry: date,
    right: Right,
    spot: float,
    T: float,
    risk_free_rate: float,
) -> OptionQuote:
    # NSE's equities endpoint has been observed using "bidprice" (lowercase
    # p) where the indices endpoint uses "bidPrice" -- check both.
    bid = _to_float(leg.get("bidprice"))
    if bid is None:
        bid = _to_float(leg.get("bidPrice"))
    ask = _to_float(leg.get("askPrice"))
    last = _to_float(leg.get("lastPrice"))
    oi = _to_float(leg.get("openInterest"))
    volume = _to_float(leg.get("totalTradedVolume"))

    nse_iv_pct = _to_float(leg.get("impliedVolatility"))
    iv = (nse_iv_pct / 100.0) if nse_iv_pct is not None and nse_iv_pct > 0 else None

    delta = gamma = theta = vega = None
    if iv is not None and iv > 0:
        g = black_scholes.greeks(
            S=spot, K=strike, T=T, sigma=iv, option_type=right, r=risk_free_rate
        )
        delta, gamma, theta, vega = g.delta, g.gamma, g.theta, g.vega

    return OptionQuote(
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


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
