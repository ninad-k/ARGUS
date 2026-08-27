"""NSE (National Stock Exchange of India) price provider.

NSE's public JSON API is unofficial, undocumented, and fronted by a WAF that
rejects bare API clients -- it requires a browser-like cookie warm-up: a GET
to ``https://www.nseindia.com/`` with a real browser ``User-Agent`` first, so
the subsequent API call can present the session cookies the WAF expects. The
warm-up + 403-retry dance itself lives in ``argus.data.nse_http.NSESession``
(shared with ``argus.options.providers.nse_options.NSEOptionsProvider``, the
options-chain sibling of this provider) -- adapted from DRUVA's
``core/options/chain_feed.py`` (``OptionChainFeed``), which independently
reimplemented the same warm-up dance for the NSE options-chain endpoint.

Only ``get_quote`` (``/api/quote-equity?symbol=``) is implemented --
``get_daily_bars`` always returns empty; NSE's historical-data endpoint is
unreliable/rate-limited enough that we don't rely on it (the composite
provider is expected to fall through to yfinance for history). IN_NSE only.

NSE's WAF may reject requests from data-center IPs regardless of headers
(observed during development) -- this is an accepted, documented limitation
of an unofficial API, not a bug in the warm-up logic itself.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime
from typing import Any

import httpx
import numpy as np
import structlog
from numpy.typing import NDArray

from argus.config import get_settings
from argus.data.nse_http import NSESession
from argus.data.prices.base import BAR_DTYPE, ProviderHealth, Quote
from argus.markets import IN_NSE, Instrument, Market

logger = structlog.get_logger(__name__)

_QUOTE_PATH = "/api/quote-equity"
_ALL_INDICES_PATH = "/api/allIndices"
_NIFTY_50_INDEX_NAME = "NIFTY 50"


class NSEProvider:
    """NSE public API quotes. IN_NSE only. Never raises.

    ``http`` is exposed for tests (inject a client wired to
    ``httpx.MockTransport``); production callers can omit it and let the
    provider own a fresh ``httpx.AsyncClient`` for its lifetime, matching the
    ``owns_client`` pattern used by ``argus.advisor.llm`` backends.
    """

    name = "nse"

    def __init__(
        self,
        *,
        timeout_seconds: float | None = None,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        data_settings = get_settings().data
        self._timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else data_settings.provider_timeout_seconds
        )
        self._session = NSESession(timeout_seconds=self._timeout_seconds, http=http)

    def supports(self, market: Market) -> bool:
        return market.code == IN_NSE.code

    async def aclose(self) -> None:
        await self._session.aclose()

    async def get_daily_bars(
        self, inst: Instrument, start: date, end: date
    ) -> NDArray[np.void]:
        # NSE's historical-data API is unreliable enough that we don't rely
        # on it -- see module docstring. Documented, not a bug.
        return np.zeros(0, dtype=BAR_DTYPE)

    async def get_quote(self, inst: Instrument) -> Quote | None:
        if inst.market_code != IN_NSE.code:
            return None
        data = await self._safe_get_json(_QUOTE_PATH, {"symbol": inst.symbol})
        if data is None:
            return None
        price_info = data.get("priceInfo") or {}
        price = _to_float(price_info.get("lastPrice"))
        if price is None:
            return None
        return Quote(
            symbol=inst.symbol,
            market_code=inst.market_code,
            price=price,
            prev_close=_to_float(price_info.get("previousClose")),
            ts=datetime.now(UTC),
        )

    async def health_check(self) -> ProviderHealth:
        data = await self._safe_get_json(_ALL_INDICES_PATH)
        if data is None:
            return ProviderHealth(
                ok=False, detail="allIndices request failed", checked_at=datetime.now(UTC)
            )
        indices = data.get("data") or []
        nifty = next(
            (row for row in indices if row.get("index") == _NIFTY_50_INDEX_NAME), None
        )
        last = _to_float(nifty.get("last")) if nifty else None
        ok = last is not None
        detail = f"NIFTY 50 last={last}" if ok else "NIFTY 50 not found in allIndices response"
        return ProviderHealth(ok=ok, detail=detail, checked_at=datetime.now(UTC))

    # ------------------------------------------------------------ internal

    async def _safe_get_json(
        self, path: str, params: dict[str, str] | None = None
    ) -> dict[str, Any] | None:
        try:
            return await asyncio.wait_for(
                self._session.get_json(path, params), timeout=self._timeout_seconds
            )
        except Exception as exc:  # noqa: BLE001 -- provider methods never raise
            logger.warning("nse.request.error", path=path, error=str(exc) or "timed out")
            return None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
