"""Pick-outcome and paper-attribution analytics endpoints (Task 14).

Both routes are pure reads over ``argus.analysis`` -- nothing here mutates a
pick or an order.
"""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Query

from argus.analysis.attribution import attribution_summary, paper_attribution
from argus.analysis.outcomes import evaluate_run_history, summarize_outcomes
from argus.api.schemas import (
    AttributionResponse,
    AttributionRowOut,
    OutcomesResponse,
    PickOutcomeOut,
)
from argus.config import get_settings
from argus.data.store.duckdb_ohlcv import BarStore

router = APIRouter(prefix="/api/v1/history", tags=["history"])

_DEFAULT_LIMIT_RUNS = 30


@router.get("/outcomes", response_model=OutcomesResponse)
async def get_outcomes(
    market: str | None = Query(default=None),
    limit_runs: int = Query(default=_DEFAULT_LIMIT_RUNS),
) -> OutcomesResponse:
    settings = get_settings()
    with BarStore(settings.duckdb_path) as store:
        outcomes = await evaluate_run_history(
            market, store, limit_runs=limit_runs, settings=settings
        )
    summary = summarize_outcomes(outcomes)
    return OutcomesResponse(
        outcomes=[PickOutcomeOut(**asdict(o)) for o in outcomes],
        summary=summary,
    )


@router.get("/attribution", response_model=AttributionResponse)
async def get_attribution() -> AttributionResponse:
    settings = get_settings()
    rows = await paper_attribution(settings=settings)
    summary = attribution_summary(rows)
    return AttributionResponse(
        rows=[AttributionRowOut(**r) for r in rows],
        summary=summary,
    )
