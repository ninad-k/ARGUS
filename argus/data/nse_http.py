"""Shared NSE cookie warm-up + HTTP GET-with-403-retry helper.

NSE's public JSON API is unofficial, undocumented, and fronted by a WAF that
rejects bare API clients -- it requires a browser-like cookie warm-up: a GET
to ``https://www.nseindia.com/`` with a real browser ``User-Agent`` first, so
the subsequent API call can present the session cookies the WAF expects.

Originally embedded in ``argus.data.prices.nse_provider.NSEProvider``
(equity quotes); factored out here so
``argus.options.providers.nse_options.NSEOptionsProvider`` (option chains)
can reuse the exact same warm-up dance instead of duplicating it. Both
providers still own their own ``NSESession`` instance -- this only shares the
*logic*, not process-wide cookie state.

Adapted from DRUVA's ``core/options/chain_feed.py`` (``OptionChainFeed``),
which independently reimplemented the same warm-up dance for the NSE
options-chain endpoint.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import structlog

logger = structlog.get_logger(__name__)

NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
    "Connection": "keep-alive",
}

NSE_BASE_URL = "https://www.nseindia.com"


class NSESession:
    """Owns an ``httpx.AsyncClient`` plus NSE cookie warm-up state.

    ``http`` is exposed for tests (inject a client wired to
    ``httpx.MockTransport``); production callers can omit it and let the
    session own a fresh ``httpx.AsyncClient`` for its lifetime -- mirrors the
    ``owns_client`` pattern used by ``argus.advisor.llm`` backends.
    """

    def __init__(
        self, *, timeout_seconds: float, http: httpx.AsyncClient | None = None
    ) -> None:
        self._timeout_seconds = timeout_seconds
        self._http = http
        self._owns_client = http is None
        self._warmed_up = False
        self._warmup_lock = asyncio.Lock()

    async def aclose(self) -> None:
        if self._owns_client and self._http is not None and not self._http.is_closed:
            await self._http.aclose()

    async def client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(
                headers=NSE_HEADERS, follow_redirects=True, timeout=self._timeout_seconds
            )
        return self._http

    async def get_json(self, path: str, params: dict[str, str] | None = None) -> dict[str, Any]:
        """GET ``path`` as JSON, warming up cookies first (once) and retrying
        once after a fresh warm-up on a 403 (a stale/rejected cookie)."""
        await self._ensure_warm()
        client = await self.client()
        resp = await client.get(f"{NSE_BASE_URL}{path}", params=params, headers=NSE_HEADERS)
        if resp.status_code == 403:
            self._warmed_up = False
            await self._ensure_warm()
            resp = await client.get(f"{NSE_BASE_URL}{path}", params=params, headers=NSE_HEADERS)
        resp.raise_for_status()
        result: dict[str, Any] = resp.json()
        return result

    async def _ensure_warm(self) -> None:
        """Fetch the NSE homepage once to obtain the cookies the API needs.

        Guarded by a lock so concurrent callers don't all warm up at once.
        Best-effort: a failed warm-up still lets the subsequent API call be
        attempted (and fail on its own, logged by the caller).
        """
        if self._warmed_up:
            return
        async with self._warmup_lock:
            if self._warmed_up:
                return
            client = await self.client()
            try:
                await client.get(f"{NSE_BASE_URL}/", headers=NSE_HEADERS)
                self._warmed_up = True
            except Exception as exc:  # noqa: BLE001 -- warm-up failure isn't fatal on its own
                logger.warning("nse.warmup.failed", error=str(exc))
