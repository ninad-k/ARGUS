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
from argus.db.models import DailyPick, ScreenRun
from argus.markets import all_markets

_RECENT_RUNS_LIMIT = 20


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
