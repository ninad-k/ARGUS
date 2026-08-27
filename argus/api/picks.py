"""Picks and screen-run endpoints.

Thin wrappers over the DB (``argus.db``) and the pipeline
(``argus.pipeline.run_daily_pipeline``). ``POST /screen/run`` runs the full
daily pipeline synchronously and returns once it completes -- with live data
sources this can take from seconds to several minutes depending on universe
size, and there is no background task queue in Phase 1, so callers must be
prepared for a slow response.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from argus.api.schemas import (
    DailyPickOut,
    LatestPicksEntry,
    LatestPicksResponse,
    RunsListResponse,
    ScreenRunOut,
    ScreenRunRequest,
    ScreenRunResponse,
)
from argus.config import get_settings
from argus.db import async_session
from argus.db.models import DailyPick, ScreenRun
from argus.markets import all_markets
from argus.pipeline import run_daily_pipeline

router = APIRouter(prefix="/api/v1", tags=["picks"])

_RECENT_RUNS_LIMIT = 20


def _run_out(run: ScreenRun) -> ScreenRunOut:
    return ScreenRunOut(
        id=run.id,
        market=run.market,
        run_ts=run.run_ts,
        universe_size=run.universe_size,
        strategies_json=run.strategies_json,
        status=run.status,
        duration_ms=run.duration_ms,
    )


def _pick_out(pick: DailyPick) -> DailyPickOut:
    return DailyPickOut(
        id=pick.id,
        run_id=pick.run_id,
        symbol=pick.symbol,
        market=pick.market,
        strategy=pick.strategy,
        score=pick.score,
        stage=pick.stage,
        reason=pick.reason,
        entry=pick.entry,
        stop=pick.stop,
        target=pick.target,
        features_json=pick.features_json,
        llm_verdict_json=pick.llm_verdict_json,
        created_at=pick.created_at,
    )


async def _latest_run_for_market(session: AsyncSession, market_code: str) -> ScreenRun | None:
    result = await session.execute(
        select(ScreenRun)
        .where(ScreenRun.market == market_code)
        .order_by(ScreenRun.run_ts.desc(), ScreenRun.id.desc())
        .limit(1)
    )
    return result.scalars().first()


@router.get("/picks/latest", response_model=LatestPicksResponse)
async def get_latest_picks(market: str | None = Query(default=None)) -> LatestPicksResponse:
    """Latest ``ScreenRun`` + its ``DailyPick`` rows, per market.

    With ``market`` given, only that market is considered (0 or 1 entries in
    the response); omitted, every market with at least one run is included.
    """
    settings = get_settings()
    market_codes = [market] if market is not None else [m.code for m in all_markets()]

    entries: list[LatestPicksEntry] = []
    async with async_session(settings) as session:
        for code in market_codes:
            run = await _latest_run_for_market(session, code)
            if run is None:
                continue
            picks_result = await session.execute(
                select(DailyPick)
                .where(DailyPick.run_id == run.id)
                .order_by(DailyPick.score.desc())
            )
            picks = list(picks_result.scalars().all())
            entries.append(
                LatestPicksEntry(run=_run_out(run), picks=[_pick_out(p) for p in picks])
            )

    return LatestPicksResponse(markets=entries)


@router.get("/runs", response_model=RunsListResponse)
async def get_recent_runs() -> RunsListResponse:
    """The most recent screen runs across all markets, newest first."""
    settings = get_settings()
    async with async_session(settings) as session:
        result = await session.execute(
            select(ScreenRun)
            .order_by(ScreenRun.run_ts.desc(), ScreenRun.id.desc())
            .limit(_RECENT_RUNS_LIMIT)
        )
        runs = list(result.scalars().all())
    return RunsListResponse(runs=[_run_out(r) for r in runs])


@router.post("/screen/run", response_model=ScreenRunResponse)
async def post_screen_run(body: ScreenRunRequest) -> ScreenRunResponse:
    """Run the full daily pipeline for ``body.market_code`` and return a summary.

    Blocks for the duration of the run -- see module docstring.
    """
    try:
        report = await run_daily_pipeline(body.market_code, top_n=body.top_n)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return ScreenRunResponse(
        run_id=report.run_id,
        market_code=report.result.market_code,
        universe_size=report.result.universe_size,
        filtered_size=report.result.filtered_size,
        top_count=len(report.result.top),
        bars_refreshed=report.bars_refreshed,
        symbols_failed=report.symbols_failed,
        llm_used=report.llm_used,
    )
