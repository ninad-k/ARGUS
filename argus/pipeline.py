"""Daily pipeline orchestration.

``run_daily_pipeline`` is the single entry point that ties every other layer
together for one market: sync the universe, refresh bars, run the screener,
optionally get an LLM review, and persist the result. It is what the
scheduler jobs (``argus.jobs.scheduler``) and the smoke-run CLI
(``scripts/smoke_daily.py``) both call.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import structlog

from argus.advisor import (
    apply_verdicts,
    build_backend,
    council_review,
    council_to_pick_verdicts,
    get_personas,
    review_picks,
)
from argus.config import AppSettings, get_settings
from argus.config.data import DataSettings
from argus.data.fundamentals import FundamentalsProvider, build_default_fundamentals
from argus.data.prices.base import PriceDataProvider, aclose_if_closeable
from argus.data.sources import (
    build_composite_from_db,
    ensure_default_sources,
    resolve_universe_provider,
)
from argus.data.store.duckdb_ohlcv import BarStore, refresh_bars, refresh_intraday
from argus.data.universe import StaticUniverseProvider, UniverseProvider, sync_instruments_to_db
from argus.db import init_db
from argus.markets import Instrument, Market, get_market
from argus.options.models import OptionChain
from argus.options.providers.base import OptionChainProvider
from argus.options.providers.factory import build_option_provider
from argus.options.suggester import (
    DerivativeSuggestion,
    ProviderFactory,
    RiskLevel,
    persist_suggestions,
    suggest_for_picks,
)
from argus.orderflow.features import compute_orderflow, to_feature_dict
from argus.screener.base import Candidate
from argus.screener.runner import ScreenResult, persist_screen_result, run_screen

logger = structlog.get_logger(__name__)

# Cap concurrent bar-refresh requests so a large universe doesn't hammer the
# upstream provider (or exhaust its own connection pool) all at once.
_REFRESH_CONCURRENCY = 4

# Extra slack (seconds) added on top of a provider's own timeout/retry budget
# for the outer per-symbol guard in ``_refresh_all`` -- see
# ``_per_symbol_refresh_timeout``. A module-level constant (rather than
# inline) so tests can shrink it to make the outer-timeout path fast to test.
_PER_SYMBOL_TIMEOUT_SLACK_SECONDS = 30.0

# Interval/lookback the post-screen orderflow-annotation step (Task 13)
# refreshes intraday bars at -- a finer grid than daily bars gives a better
# volume profile without pulling a whole universe's worth of intraday data
# (only run for ``result.top``, see ``_annotate_orderflow``).
_ORDERFLOW_INTRADAY_INTERVAL = "15m"
_ORDERFLOW_INTRADAY_LOOKBACK_DAYS = 30


def _per_symbol_refresh_timeout(data_settings: DataSettings) -> float:
    """Belt-and-braces outer timeout for one symbol's bar refresh.

    ``YFinanceProvider`` (and any well-behaved ``PriceDataProvider``) already
    bounds its own calls with ``provider_timeout_seconds`` and retries up to
    ``provider_max_retries`` times, but nothing stops a *future* provider
    implementation from ignoring that budget and hanging forever. This outer
    ``asyncio.wait_for`` in ``_refresh_all._one`` is a generous guard on top
    of the provider's own budget so a single hung call can never block the
    whole ``asyncio.gather`` -- and therefore the scheduled daily run -- from
    completing.
    """
    return (
        data_settings.provider_timeout_seconds * (data_settings.provider_max_retries + 1)
        + _PER_SYMBOL_TIMEOUT_SLACK_SECONDS
    )


@dataclass
class ScreenReport:
    """The full outcome of one ``run_daily_pipeline`` call."""

    result: ScreenResult
    run_id: int
    bars_refreshed: int
    symbols_failed: list[str]
    llm_used: bool
    # Derivative ideas (Task 12) keyed by symbol, for whichever of
    # ``result.top`` had one -- empty when ``settings.options.enabled`` is
    # off, no pick's instrument has options/futures, or the suggestion step
    # failed (it never raises into the pipeline, see ``_suggest_derivatives``).
    suggestions: dict[str, DerivativeSuggestion] = field(default_factory=dict)


async def run_daily_pipeline(
    market_code: str,
    *,
    top_n: int = 5,
    refresh: bool = True,
    llm: bool = True,
    provider: PriceDataProvider | None = None,
    store: BarStore | None = None,
    universe_provider: UniverseProvider | None = None,
    option_provider_factory: ProviderFactory | None = None,
    fundamentals_provider: FundamentalsProvider | None = None,
) -> ScreenReport:
    """Run the full daily pipeline for ``market_code`` and persist the result.

    Steps: ensure the control-plane DB/default data sources exist -> resolve
    the universe and sync it to the DB -> (optionally) refresh OHLCV bars for
    every instrument, with bounded concurrency and per-symbol failure
    tolerance -> run the screener -> (optionally) get an LLM review of the
    top picks -> persist the ``ScreenRun``/``DailyPick`` rows -> (optionally)
    attach derivative suggestions to the top picks.

    ``provider``/``store``/``universe_provider`` are injection points for
    tests and the smoke-run script; when omitted, production defaults are
    built from settings. A ``store``/``provider`` this call creates itself is
    closed before returning (``provider`` only if it implements
    ``CloseablePriceProvider``, e.g. a ``CompositePriceProvider`` fanning out
    to an ``NSEProvider``); a caller-supplied ``store``/``provider`` is left
    open for the caller to manage. ``option_provider_factory`` is the same
    kind of injection point for ``argus.options.suggester.suggest_for_picks``
    -- tests pass a factory serving a ``StaticOptionChainProvider`` instead
    of hitting a live options data source.
    """
    settings = get_settings()
    market = get_market(market_code)

    await init_db(settings)
    await ensure_default_sources(settings)

    owns_store = store is None
    owns_provider = provider is None
    resolved_store = store if store is not None else BarStore(settings.duckdb_path)
    resolved_provider = (
        provider if provider is not None else await build_composite_from_db(settings)
    )
    resolved_universe: UniverseProvider = (
        universe_provider
        if universe_provider is not None
        else await resolve_universe_provider(market_code, settings)
    )

    try:
        # Resolved once here rather than a second time inside ``run_screen`` --
        # with a live (uncached) universe provider like ``TVUniverseProvider``,
        # calling ``.universe()`` twice can return different results across the
        # two live scanner queries, or silently fall back to seed CSVs on the
        # second call even though bars were only refreshed for the first
        # call's instruments. ``run_screen`` gets the concrete list wrapped in
        # a ``StaticUniverseProvider`` so its own ``.universe()`` call is free.
        instruments = await resolved_universe.universe(market)
        await sync_instruments_to_db(instruments, settings)

        bars_refreshed = 0
        symbols_failed: list[str] = []
        if refresh:
            bars_refreshed, symbols_failed = await _refresh_all(
                resolved_store,
                resolved_provider,
                instruments,
                settings.data.bar_lookback_days,
                per_symbol_timeout=_per_symbol_refresh_timeout(settings.data),
            )

        result = await run_screen(
            market,
            store=resolved_store,
            universe_provider=StaticUniverseProvider({market.code: instruments}),
            top_n=top_n,
            fundamentals_provider=(
                fundamentals_provider
                if fundamentals_provider is not None
                else build_default_fundamentals()
            ),
        )

        chain_cache = await _annotate_orderflow(
            result, resolved_store, resolved_provider, settings, option_provider_factory
        )

        llm_used = False
        if llm and settings.llm.enabled and result.top:
            llm_used = await _review_with_llm(result, market, settings)

        run_id = await persist_screen_result(result, settings)

        report = ScreenReport(
            result=result,
            run_id=run_id,
            bars_refreshed=bars_refreshed,
            symbols_failed=symbols_failed,
            llm_used=llm_used,
        )

        if settings.options.enabled:
            report.suggestions = await _suggest_derivatives(
                report, settings, option_provider_factory, chain_cache
            )

        return report
    finally:
        if owns_store:
            resolved_store.close()
        if owns_provider:
            await aclose_if_closeable(resolved_provider)


async def _refresh_all(
    store: BarStore,
    provider: PriceDataProvider,
    instruments: list[Instrument],
    lookback_days: int,
    *,
    per_symbol_timeout: float,
) -> tuple[int, list[str]]:
    """Refresh bars for every instrument with bounded concurrency.

    A per-symbol failure -- including a per-symbol timeout, see
    ``per_symbol_timeout`` -- is logged and recorded, never allowed to abort
    the rest of the batch.
    """
    semaphore = asyncio.Semaphore(_REFRESH_CONCURRENCY)
    total_added = 0
    failed: list[str] = []

    async def _one(inst: Instrument) -> None:
        nonlocal total_added
        async with semaphore:
            try:
                added = await asyncio.wait_for(
                    refresh_bars(store, provider, inst, lookback_days), timeout=per_symbol_timeout
                )
            except TimeoutError:
                logger.warning(
                    "pipeline.refresh_bars.timeout",
                    symbol=inst.symbol,
                    market=inst.market_code,
                    timeout_seconds=per_symbol_timeout,
                )
                failed.append(inst.symbol)
                return
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


async def _annotate_orderflow(
    result: ScreenResult,
    store: BarStore,
    provider: PriceDataProvider,
    settings: AppSettings,
    option_provider_factory: ProviderFactory | None,
) -> dict[tuple[str, str], OptionChain]:
    """Orderflow annotation step (Task 13) for ``result.top`` only -- never
    the whole universe, keeping this cheap.

    Per top pick: refresh intraday bars (a no-op when the provider has no
    intraday support, e.g. every provider but ``YFinanceProvider`` today --
    see ``PriceDataProvider.get_intraday_bars``), fetch its option chain
    once (when options are enabled and the instrument has options/futures),
    and compute ``OrderflowFeatures`` from stored daily + intraday bars +
    that chain, stashing the result under
    ``Candidate.features["orderflow"]`` so it round-trips through
    ``persist_screen_result`` into ``DailyPick.features_json``.

    The fetched chain is returned (keyed by ``(symbol, market_code)``) so
    ``_suggest_derivatives``/``suggest_for_picks`` can reuse it instead of
    fetching the same chain again -- see ``suggest_for_picks``'s
    ``chain_cache`` parameter. Exception-contained per pick: a bad
    refresh/fetch/compute never drops a pick or breaks the pipeline, it just
    leaves that pick without an "orderflow" key (and/or a cached chain).
    """
    resolved_factory = (
        option_provider_factory if option_provider_factory is not None else build_option_provider
    )

    chain_cache: dict[tuple[str, str], OptionChain] = {}

    async def _one(candidate: Candidate) -> None:
        inst = candidate.instrument

        try:
            await refresh_intraday(
                store,
                provider,
                inst,
                interval=_ORDERFLOW_INTRADAY_INTERVAL,
                lookback_days=_ORDERFLOW_INTRADAY_LOOKBACK_DAYS,
            )
        except Exception as exc:  # noqa: BLE001 -- must never break the pipeline
            logger.warning(
                "pipeline.orderflow.intraday_refresh_failed", symbol=inst.symbol, error=str(exc)
            )

        chain: OptionChain | None = None
        if settings.options.enabled and (inst.has_options or inst.has_futures):
            option_provider: OptionChainProvider | None = None
            try:
                option_provider = resolved_factory(inst)
                if option_provider is not None:
                    chain = await option_provider.get_chain(inst)
                    if chain is not None:
                        chain_cache[(inst.symbol, inst.market_code)] = chain
            except Exception as exc:  # noqa: BLE001 -- must never break the pipeline
                logger.warning(
                    "pipeline.orderflow.chain_fetch_failed", symbol=inst.symbol, error=str(exc)
                )
            finally:
                if option_provider is not None:
                    try:
                        await option_provider.aclose()
                    except Exception as exc:  # noqa: BLE001 -- must never break the pipeline
                        logger.warning(
                            "pipeline.orderflow.chain_aclose_failed",
                            symbol=inst.symbol,
                            error=str(exc),
                        )

        try:
            daily = await asyncio.to_thread(
                store.get_bars, inst.market_code, inst.symbol, 260
            )
            intraday = await asyncio.to_thread(
                store.get_intraday,
                inst.market_code,
                inst.symbol,
                _ORDERFLOW_INTRADAY_INTERVAL,
            )
            of = compute_orderflow(
                daily,
                intraday_bars=intraday if len(intraday) > 0 else None,
                chain=chain if inst.has_options else None,
            )
            if of is not None:
                candidate.features["orderflow"] = to_feature_dict(of)
        except Exception as exc:  # noqa: BLE001 -- must never break the pipeline
            logger.warning("pipeline.orderflow.compute_failed", symbol=inst.symbol, error=str(exc))

    try:
        await asyncio.gather(*(_one(c) for c in result.top))
    except Exception as exc:  # noqa: BLE001 -- must never break the pipeline
        logger.warning("pipeline.orderflow.annotate_failed", error=str(exc))
    return chain_cache


async def _suggest_derivatives(
    report: ScreenReport,
    settings: AppSettings,
    option_provider_factory: ProviderFactory | None,
    chain_cache: dict[tuple[str, str], OptionChain] | None = None,
) -> dict[str, DerivativeSuggestion]:
    """Attach derivative suggestions to ``report``'s top picks (Task 12).

    ``chain_cache`` is the (symbol, market_code)-keyed chain cache
    ``_annotate_orderflow`` (Task 13) built while annotating the same picks
    -- passed through to ``suggest_for_picks`` so a pick's chain is fetched
    at most once per pipeline run.

    Exception-contained: a bad risk-level string, a suggester bug, or a DB
    hiccup while persisting degrades to "no suggestions" rather than
    breaking the pipeline -- ``suggest_for_picks``/``persist_suggestions``
    already never raise on their own, this is belt-and-braces around them
    plus the ``RiskLevel(...)`` parse.
    """
    try:
        risk = RiskLevel(settings.options.risk_level)
    except ValueError:
        logger.warning(
            "pipeline.option_suggestions.bad_risk_level", risk_level=settings.options.risk_level
        )
        risk = RiskLevel.MODERATE

    try:
        suggestions = await suggest_for_picks(
            report,
            risk=risk,
            settings=settings.options,
            provider_factory=option_provider_factory,
            chain_cache=chain_cache,
        )
        if suggestions:
            await persist_suggestions(report.run_id, suggestions, risk, settings)
        return suggestions
    except Exception as exc:  # noqa: BLE001 -- must never break the pipeline
        logger.warning("pipeline.option_suggestions.failed", error=str(exc))
        return {}


async def _review_with_llm(result: ScreenResult, market: Market, settings: AppSettings) -> bool:
    """Run the LLM review over ``result.top`` and apply verdicts in place.

    Returns whether any verdict was actually produced. A backend construction
    or review failure degrades to "no verdicts" rather than raising.

    ``build_backend`` may own an ``httpx.AsyncClient`` (see
    ``argus.advisor.llm``) -- this runs once per daily pipeline call inside a
    long-lived scheduler process, so the backend is always closed via
    ``aclose()`` once the review is done, success or failure.

    When ``settings.llm.council_enabled``, this fans the review out to the
    configured investor personas (``argus.advisor.council``) instead of the
    single-pass reviewer -- one LLM call per persona instead of one call
    total, deterministically fused into the same ``PickVerdict`` shape so
    everything downstream (persistence, UI) is unaffected either way.
    """
    try:
        backend = build_backend(settings.llm)
    except Exception as exc:  # LLM/backend failure must never break the pipeline
        logger.warning("pipeline.llm_review.failed", error=str(exc))
        return False

    try:
        if settings.llm.council_enabled:
            slugs = [s.strip() for s in settings.llm.council_personas.split(",") if s.strip()]
            personas = get_personas(slugs)
            council = await council_review(result.top, market, backend, personas)
            verdicts = council_to_pick_verdicts(council)
        else:
            verdicts = await review_picks(result.top, market, backend)
    except Exception as exc:  # LLM/backend failure must never break the pipeline
        logger.warning("pipeline.llm_review.failed", error=str(exc))
        return False
    finally:
        await backend.aclose()

    apply_verdicts(result, verdicts)
    return bool(verdicts)
