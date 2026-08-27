#!/usr/bin/env python
"""Smoke-run the daily pipeline for one market and print/save its report.

Default mode is fully offline: an in-memory ``StaticPriceProvider`` seeded
with ~8 synthetic symbols (varied trend/flat/breakout shapes) and a fixed
``StaticUniverseProvider`` stand in for the real universe/provider, so the
whole pipeline can be exercised with no network access. ``--live`` swaps
those for the real seed universe (sliced to ~20 symbols) and the default
composite (yfinance-backed) provider.

All actual logic lives in ``argus`` modules -- this script only wires
together injected dependencies and prints the result.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import date, timedelta
from pathlib import Path

import httpx

from argus.config import get_settings
from argus.data.fundamentals import FundamentalsProvider, NullFundamentalsProvider
from argus.data.prices.base import PriceDataProvider
from argus.data.prices.static_provider import StaticPriceProvider, synthetic_bars
from argus.data.store.duckdb_ohlcv import BarStore
from argus.data.universe import SeedUniverseProvider, StaticUniverseProvider, UniverseProvider
from argus.markets import Instrument, Market, get_market
from argus.paper.engine import run_paper_cycle
from argus.pipeline import run_daily_pipeline
from argus.reports import render_markdown_report, save_report

_LIVE_UNIVERSE_LIMIT = 20
_OLLAMA_PROBE_TIMEOUT_S = 2.0


class _SlicedUniverseProvider:
    """Wraps another ``UniverseProvider`` and truncates its result.

    Used only by ``--live`` mode, to keep a smoke run's yfinance calls small
    instead of pulling the full seed universe (hundreds of symbols).
    """

    name = "sliced"

    def __init__(self, inner: UniverseProvider, limit: int) -> None:
        self._inner = inner
        self._limit = limit

    async def universe(self, market: Market) -> list[Instrument]:
        instruments = await self._inner.universe(market)
        return instruments[: self._limit]


def _build_offline_universe(market_code: str) -> tuple[StaticPriceProvider, StaticUniverseProvider]:
    """~8 synthetic symbols spanning uptrend/downtrend/flat/breakout/reject shapes."""
    today = date.today()  # noqa: DTZ011 -- matches refresh_bars' own daily-cache boundary
    n = 300
    start = today - timedelta(days=n - 1)

    specs: list[tuple[str, float, int, float]] = [
        # symbol, start_price, seed, trend
        ("SMOKE_UP1", 100.0, 1, 0.005),
        ("SMOKE_UP2", 50.0, 2, 0.006),
        ("SMOKE_FLAT", 80.0, 3, 0.0),
        ("SMOKE_DOWN", 120.0, 4, -0.004),
        ("SMOKE_MILD", 60.0, 5, 0.002),
        ("SMOKE_PENNY", 2.0, 6, 0.002),  # rejected: below min-price filter
        ("SMOKE_THIN", 90.0, 7, 0.001),  # rejected: below min-volume filter
    ]

    provider = StaticPriceProvider()
    for symbol, start_price, seed, trend in specs:
        bars = synthetic_bars(n=n, start_price=start_price, seed=seed, start=start, trend=trend)
        if symbol == "SMOKE_THIN":
            bars["volume"] = 50_000.0  # deterministic low-liquidity rejection
        provider.add(symbol, bars)

    # Short history -> rejected by the min-history filter.
    short_start = today - timedelta(days=49)
    provider.add(
        "SMOKE_SHORT",
        synthetic_bars(n=50, start_price=100.0, seed=8, start=short_start, trend=0.001),
    )

    instruments = [Instrument(symbol=symbol, market_code=market_code) for symbol, *_ in specs]
    instruments.append(Instrument(symbol="SMOKE_SHORT", market_code=market_code))

    universe_provider = StaticUniverseProvider({market_code: instruments})
    return provider, universe_provider


async def _ollama_reachable(base_url: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=_OLLAMA_PROBE_TIMEOUT_S) as client:
            resp = await client.get(f"{base_url.rstrip('/')}/api/tags")
            return resp.status_code == 200
    except Exception:
        return False


async def _run(args: argparse.Namespace) -> int:
    settings = get_settings()
    market = get_market(args.market)

    llm_enabled = not args.no_llm
    if llm_enabled and settings.llm.provider == "ollama":
        llm_enabled = await _ollama_reachable(settings.llm.base_url)

    provider: PriceDataProvider
    universe_provider: UniverseProvider
    store: BarStore
    fundamentals_provider: FundamentalsProvider | None = None

    if args.live:
        from argus.data.sources import build_composite_from_db, ensure_default_sources

        await ensure_default_sources(settings)
        provider = await build_composite_from_db(settings)
        universe_provider = _SlicedUniverseProvider(SeedUniverseProvider(), _LIVE_UNIVERSE_LIMIT)
        store = BarStore(settings.duckdb_path)
    else:
        provider, universe_provider = _build_offline_universe(market.code)
        store = BarStore(settings.data_dir / "smoke.duckdb")
        # Offline mode must stay offline: the pipeline's default fundamentals
        # provider (yfinance) would otherwise issue live lookups for the
        # synthetic SMOKE_* symbols.
        fundamentals_provider = NullFundamentalsProvider()

    try:
        report = await run_daily_pipeline(
            args.market,
            top_n=args.top,
            refresh=True,
            llm=llm_enabled,
            provider=provider,
            store=store,
            universe_provider=universe_provider,
            fundamentals_provider=fundamentals_provider,
        )
        if args.paper:
            await run_paper_cycle(args.market, report, store)
    finally:
        store.close()

    markdown = render_markdown_report(report)
    print(markdown)

    saved_path = save_report(report, out_dir=args.out)
    print(f"Report saved to: {saved_path}", file=sys.stderr)
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-run the ARGUS daily pipeline.")
    parser.add_argument(
        "--market", required=True, choices=["US_NASDAQ", "US_NYSE", "IN_NSE"], help="Market code."
    )
    parser.add_argument("--top", type=int, default=5, help="Number of top picks to keep.")
    parser.add_argument(
        "--live", action="store_true", help="Use real yfinance data instead of synthetic bars."
    )
    parser.add_argument(
        "--no-llm", action="store_true", help="Skip the LLM review even if a backend is reachable."
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Also exercise the paper-trading cycle (fill/exit/queue/snapshot) offline.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Directory to save the report in.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        return asyncio.run(_run(args))
    except Exception as exc:  # a smoke run must report failure clearly, not traceback-spam
        print(f"smoke_daily failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
