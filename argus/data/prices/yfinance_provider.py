"""Yahoo Finance price provider (via the ``yfinance`` package).

``yfinance`` is a synchronous, blocking library — every call here is pushed
onto a thread via ``asyncio.to_thread`` so it doesn't block the event loop.
Provider methods never raise: any failure is logged and an empty/``None``
result is returned instead, so a flaky upstream never takes down a screen run.
"""

import asyncio
from datetime import UTC, date, datetime, timedelta

import numpy as np
import structlog
import yfinance as yf
from numpy.typing import NDArray

from argus.config import get_settings
from argus.data.prices.base import BAR_DTYPE, ProviderHealth, Quote, bars_from_columns
from argus.markets import Instrument, Market
from argus.markets.registry import IN_NSE
from argus.utils.retry import retry_async

logger = structlog.get_logger(__name__)


def _yahoo_ticker(inst: Instrument) -> str:
    """Map an ``Instrument`` to its Yahoo Finance ticker symbol."""
    if inst.market_code == IN_NSE.code:
        return f"{inst.symbol}.NS"
    return inst.symbol


class YFinanceProvider:
    """Daily OHLCV + quotes sourced from Yahoo Finance. Supports all markets."""

    name = "yfinance"

    def __init__(
        self, *, timeout_seconds: float | None = None, max_retries: int | None = None
    ) -> None:
        data_settings = get_settings().data
        self._timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else data_settings.provider_timeout_seconds
        )
        self._max_retries = (
            max_retries if max_retries is not None else data_settings.provider_max_retries
        )

    def supports(self, market: Market) -> bool:
        return True

    async def get_daily_bars(
        self, inst: Instrument, start: date, end: date
    ) -> NDArray[np.void]:
        def _on_error(attempt: int, exc: BaseException) -> None:
            logger.warning(
                "yfinance.get_daily_bars.attempt_failed",
                symbol=inst.symbol,
                market=inst.market_code,
                attempt=attempt,
                # a bare TimeoutError means the call ran past our own
                # timeout, not that yfinance itself raised
                error=str(exc) or "timed out",
            )

        try:
            return await retry_async(
                lambda: asyncio.to_thread(self._fetch_daily_bars, inst, start, end),
                attempts=self._max_retries + 1,
                timeout_seconds=self._timeout_seconds,
                retry_if=lambda bars: len(bars) == 0,
                on_error=_on_error,
            )
        except Exception as exc:  # noqa: BLE001 — provider methods never raise
            logger.warning(
                "yfinance.get_daily_bars.error",
                symbol=inst.symbol,
                market=inst.market_code,
                error=str(exc),
            )
            return np.zeros(0, dtype=BAR_DTYPE)

    def _fetch_daily_bars(self, inst: Instrument, start: date, end: date) -> NDArray[np.void]:
        ticker = yf.Ticker(_yahoo_ticker(inst))
        # yfinance's `end` is exclusive — add a day so callers can pass an
        # inclusive end date.
        hist = ticker.history(
            start=start,
            end=end + timedelta(days=1),
            interval="1d",
            auto_adjust=False,
        )
        if hist is None or hist.empty:
            return np.zeros(0, dtype=BAR_DTYPE)

        hist = hist.dropna(subset=["Open", "High", "Low", "Close"])
        if hist.empty:
            return np.zeros(0, dtype=BAR_DTYPE)

        index = hist.index
        if getattr(index, "tz", None) is not None:
            index = index.tz_localize(None)
        ts = index.to_numpy().astype("datetime64[s]")

        return bars_from_columns(
            ts,
            hist["Open"].to_numpy(dtype=np.float64),
            hist["High"].to_numpy(dtype=np.float64),
            hist["Low"].to_numpy(dtype=np.float64),
            hist["Close"].to_numpy(dtype=np.float64),
            hist["Volume"].to_numpy(dtype=np.float64),
        )

    async def get_quote(self, inst: Instrument) -> Quote | None:
        # A single attempt -- quotes are non-critical (the screener falls
        # back to the last daily close), so unlike get_daily_bars this isn't
        # worth retrying. Still timeout-bounded so a hung call can't block
        # the caller (see get_daily_bars/_per_symbol_refresh_timeout).
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._fetch_quote, inst), timeout=self._timeout_seconds
            )
        except Exception as exc:  # noqa: BLE001 — provider methods never raise
            logger.warning(
                "yfinance.get_quote.error",
                symbol=inst.symbol,
                market=inst.market_code,
                error=str(exc) or "timed out",
            )
            return None

    def _fetch_quote(self, inst: Instrument) -> Quote | None:
        ticker = yf.Ticker(_yahoo_ticker(inst))
        try:
            fast_info = ticker.fast_info
            price = float(fast_info.last_price)
            prev_close = getattr(fast_info, "previous_close", None)
            return Quote(
                symbol=inst.symbol,
                market_code=inst.market_code,
                price=price,
                prev_close=float(prev_close) if prev_close is not None else None,
                ts=datetime.now(UTC),
            )
        except Exception as exc:  # noqa: BLE001 — fall back to last daily close
            logger.debug(
                "yfinance.get_quote.fast_info_failed",
                symbol=inst.symbol,
                error=str(exc),
            )

        today = date.today()  # noqa: DTZ011 — daily-bar boundary only, market tz irrelevant here
        bars = self._fetch_daily_bars(inst, today - timedelta(days=10), today)
        if len(bars) == 0:
            return None
        last = bars[-1]
        prev_close = float(bars[-2]["close"]) if len(bars) >= 2 else None
        ts = last["ts"].astype("datetime64[s]").astype(datetime).replace(tzinfo=UTC)
        return Quote(
            symbol=inst.symbol,
            market_code=inst.market_code,
            price=float(last["close"]),
            prev_close=prev_close,
            ts=ts,
        )

    async def health_check(self) -> ProviderHealth:
        try:
            bars = await asyncio.wait_for(
                asyncio.to_thread(self._health_check_sync), timeout=self._timeout_seconds
            )
        except Exception as exc:  # noqa: BLE001 — provider methods never raise
            return ProviderHealth(
                ok=False, detail=str(exc) or "timed out", checked_at=datetime.now(UTC)
            )

        ok = len(bars) > 0
        detail = f"fetched {len(bars)} AAPL bars" if ok else "no bars returned for AAPL"
        return ProviderHealth(ok=ok, detail=detail, checked_at=datetime.now(UTC))

    def _health_check_sync(self) -> NDArray[np.void]:
        today = date.today()  # noqa: DTZ011 — health probe only, exact tz doesn't matter
        aapl = Instrument(symbol="AAPL", market_code="US_NASDAQ")
        return self._fetch_daily_bars(aapl, today - timedelta(days=5), today)
