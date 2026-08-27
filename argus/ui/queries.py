"""Small async DB-read helpers shared by the NiceGUI pages.

Kept separate from ``argus.api`` -- the API returns Pydantic response models
over HTTP, these return ORM rows straight to Python page code. Data-source
CRUD (used by both the API and the ``/sources`` page) lives in
``argus.data.sources`` instead, so that logic exists in exactly one place.
"""

from __future__ import annotations

from sqlalchemy import select

from argus.config import AppSettings, get_settings
from argus.db import async_session
from argus.db.models import (
    DailyPick,
    OptionSuggestion,
    PaperEquityPoint,
    PaperOrder,
    PaperPosition,
    ScreenRun,
)
from argus.markets import all_markets

_RECENT_RUNS_LIMIT = 20
_RECENT_PAPER_ORDERS_LIMIT = 50


async def latest_run_for_market(
    market_code: str, settings: AppSettings | None = None
) -> ScreenRun | None:
    settings = settings or get_settings()
    async with async_session(settings) as session:
        result = await session.execute(
            select(ScreenRun)
            .where(ScreenRun.market == market_code)
            .order_by(ScreenRun.run_ts.desc(), ScreenRun.id.desc())
            .limit(1)
        )
        return result.scalars().first()


async def picks_for_run(run_id: int, settings: AppSettings | None = None) -> list[DailyPick]:
    settings = settings or get_settings()
    async with async_session(settings) as session:
        result = await session.execute(
            select(DailyPick).where(DailyPick.run_id == run_id).order_by(DailyPick.score.desc())
        )
        return list(result.scalars().all())


async def option_suggestions_for_run(
    run_id: int, settings: AppSettings | None = None
) -> dict[int, OptionSuggestion]:
    """Option suggestions for ``run_id``'s picks, keyed by ``pick_id``."""
    settings = settings or get_settings()
    async with async_session(settings) as session:
        pick_ids_result = await session.execute(
            select(DailyPick.id).where(DailyPick.run_id == run_id)
        )
        pick_ids = [pid for (pid,) in pick_ids_result.all()]
        if not pick_ids:
            return {}
        result = await session.execute(
            select(OptionSuggestion).where(OptionSuggestion.pick_id.in_(pick_ids))
        )
        return {s.pick_id: s for s in result.scalars().all()}


async def recent_runs(
    limit: int = _RECENT_RUNS_LIMIT, settings: AppSettings | None = None
) -> list[ScreenRun]:
    settings = settings or get_settings()
    async with async_session(settings) as session:
        result = await session.execute(
            select(ScreenRun).order_by(ScreenRun.run_ts.desc(), ScreenRun.id.desc()).limit(limit)
        )
        return list(result.scalars().all())


async def top_picks_across_markets(settings: AppSettings | None = None) -> list[DailyPick]:
    """Each market's latest-run picks, merged and ranked by score descending."""
    settings = settings or get_settings()
    picks: list[DailyPick] = []
    for market in all_markets():
        run = await latest_run_for_market(market.code, settings)
        if run is None:
            continue
        picks.extend(await picks_for_run(run.id, settings))
    picks.sort(key=lambda p: p.score, reverse=True)
    return picks


async def open_paper_positions(settings: AppSettings | None = None) -> list[PaperPosition]:
    settings = settings or get_settings()
    async with async_session(settings) as session:
        result = await session.execute(
            select(PaperPosition)
            .where(PaperPosition.closed_at.is_(None))
            .order_by(PaperPosition.opened_at.desc())
        )
        return list(result.scalars().all())


async def recent_paper_orders(
    limit: int = _RECENT_PAPER_ORDERS_LIMIT, settings: AppSettings | None = None
) -> list[PaperOrder]:
    settings = settings or get_settings()
    async with async_session(settings) as session:
        result = await session.execute(
            select(PaperOrder).order_by(PaperOrder.created_at.desc()).limit(limit)
        )
        return list(result.scalars().all())


async def paper_equity_points(settings: AppSettings | None = None) -> list[PaperEquityPoint]:
    settings = settings or get_settings()
    async with async_session(settings) as session:
        result = await session.execute(
            select(PaperEquityPoint).order_by(PaperEquityPoint.date.asc())
        )
        return list(result.scalars().all())


async def total_realized_pnl(settings: AppSettings | None = None) -> float:
    """Sum of realized P&L across every position (open or closed) that has
    ever had a sell fill applied to it."""
    settings = settings or get_settings()
    async with async_session(settings) as session:
        result = await session.execute(select(PaperPosition.realized_pnl))
        values: list[float] = [v for (v,) in result.all() if v is not None]
    return round(sum(values), 2)
