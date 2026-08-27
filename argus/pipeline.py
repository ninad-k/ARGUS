"""Daily pipeline orchestration.

``run_daily_pipeline`` is the single entry point that ties every other layer
together for one market: sync the universe, refresh bars, run the screener,
optionally get an LLM review, and persist the result. It is what the
scheduler jobs (``argus.jobs.scheduler``) and the smoke-run CLI
(``scripts/smoke_daily.py``) both call.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import structlog

from argus.advisor import apply_verdicts, build_backend, review_picks
from argus.config import AppSettings, get_settings
from argus.data.prices.base import PriceDataProvider
from argus.data.sources import build_composite_from_db, ensure_default_sources
from argus.data.store.duckdb_ohlcv import BarStore, refresh_bars
from argus.data.universe import SeedUniverseProvider, UniverseProvider, sync_instruments_to_db
from argus.db import init_db
from argus.markets import Instrument, Market, get_market
from argus.screener.runner import ScreenResult, persist_screen_result, run_screen

logger = structlog.get_logger(__name__)

# Cap concurrent bar-refresh requests so a large universe doesn't hammer the
# upstream provider (or exhaust its own connection pool) all at once.
_REFRESH_CONCURRENCY = 4


@dataclass
class ScreenReport:
    """The full outcome of one ``run_daily_pipeline`` call."""

    result: ScreenResult
    run_id: int
    bars_refreshed: int
    symbols_failed: list[str]
    llm_used: bool


async def run_daily_pipeline(
    market_code: str,
    *,
    top_n: int = 5,
    refresh: bool = True,
    llm: bool = True,
    provider: PriceDataProvider | None = None,
    store: BarStore | None = None,
    universe_provider: UniverseProvider | None = None,
) -> ScreenReport:
    """Run the full daily pipeline for ``market_code`` and persist the result.

    Steps: ensure the control-plane DB/default data sources exist -> resolve
    the universe and sync it to the DB -> (optionally) refresh OHLCV bars for
    every instrument, with bounded concurrency and per-symbol failure
    tolerance -> run the screener -> (optionally) get an LLM review of the
    top picks -> persist the ``ScreenRun``/``DailyPick`` rows.

    ``provider``/``store``/``universe_provider`` are injection points for
    tests and the smoke-run script; when omitted, production defaults are
    built from settings. A ``store`` this call creates itself is closed
    before returning; a caller-supplied ``store`` is left open for the
    caller to manage.
    """
    settings = get_settings()
    market = get_market(market_code)

    await init_db(settings)
    await ensure_default_sources(settings)

    owns_store = store is None
    resolved_store = store if store is not None else BarStore(settings.duckdb_path)
    resolved_provider = (
        provider if provider is not None else await build_composite_from_db(settings)
    )
    resolved_universe: UniverseProvider = (
        universe_provider if universe_provider is not None else SeedUniverseProvider()
    )

    try:
        instruments = await resolved_universe.universe(market)
        await sync_instruments_to_db(instruments, settings)

        bars_refreshed = 0
        symbols_failed: list[str] = []
        if refresh:
            bars_refreshed, symbols_failed = await _refresh_all(
                resolved_store, resolved_provider, instruments, settings.data.bar_lookback_days
            )

        result = await run_screen(
            market,
            store=resolved_store,
            universe_provider=resolved_universe,
            top_n=top_n,
        )

        llm_used = False
        if llm and settings.llm.enabled and result.top:
            llm_used = await _review_with_llm(result, market, settings)

        run_id = await persist_screen_result(result, settings)

        return ScreenReport(
            result=result,
            run_id=run_id,
            bars_refreshed=bars_refreshed,
            symbols_failed=symbols_failed,
            llm_used=llm_used,
        )
    finally:
        if owns_store:
            resolved_store.close()


async def _refresh_all(
    store: BarStore,
    provider: PriceDataProvider,
    instruments: list[Instrument],
    lookback_days: int,
) -> tuple[int, list[str]]:
    """Refresh bars for every instrument with bounded concurrency.

    A per-symbol failure is logged and recorded, never allowed to abort the
    rest of the batch.
    """
    semaphore = asyncio.Semaphore(_REFRESH_CONCURRENCY)
    total_added = 0
    failed: list[str] = []

    async def _one(inst: Instrument) -> None:
        nonlocal total_added
        async with semaphore:
            try:
                added = await refresh_bars(store, provider, inst, lookback_days)
            except Exception as exc:  # a provider hiccup must not abort the run
                logger.warning(
                    "pipeline.refresh_bars.failed",
                    symbol=inst.symbol,
                    market=inst.market_code,
                    error=str(exc),
                )
                failed.append(inst.symbol)
                return
            total_added += added

    await asyncio.gather(*(_one(inst) for inst in instruments))
    return total_added, failed


async def _review_with_llm(result: ScreenResult, market: Market, settings: AppSettings) -> bool:
    """Run the LLM review over ``result.top`` and apply verdicts in place.

    Returns whether any verdict was actually produced. A backend construction
    or review failure degrades to "no verdicts" rather than raising.
    """
    try:
        backend = build_backend(settings.llm)
        verdicts = await review_picks(result.top, market, backend)
    except Exception as exc:  # LLM/backend failure must never break the pipeline
        logger.warning("pipeline.llm_review.failed", error=str(exc))
        return False

    apply_verdicts(result, verdicts)
    return bool(verdicts)
