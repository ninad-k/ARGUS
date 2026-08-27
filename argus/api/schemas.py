"""Pydantic request/response schemas for the REST API.

These are pure DTOs -- conversion to/from the SQLAlchemy models lives in the
router modules, not here.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field


class ScreenRunOut(BaseModel):
    id: int
    market: str
    run_ts: datetime
    universe_size: int
    strategies_json: dict[str, Any]
    status: str
    duration_ms: int | None


class OptionSuggestionOut(BaseModel):
    id: int
    risk_level: str
    instrument_type: str
    strike: float
    expiry: datetime
    suggested_price: float | None
    iv: float | None
    delta: float | None
    oi: int | None
    rationale: str | None


class DailyPickOut(BaseModel):
    id: int
    run_id: int
    symbol: str
    market: str
    strategy: str
    score: float
    stage: str
    reason: str | None
    entry: float | None
    stop: float | None
    target: float | None
    features_json: dict[str, Any]
    llm_verdict_json: dict[str, Any] | None
    created_at: datetime
    option_suggestion: OptionSuggestionOut | None = None


class LatestPicksEntry(BaseModel):
    run: ScreenRunOut
    picks: list[DailyPickOut]


class LatestPicksResponse(BaseModel):
    markets: list[LatestPicksEntry]


class RunsListResponse(BaseModel):
    runs: list[ScreenRunOut]


class ScreenRunRequest(BaseModel):
    market_code: str
    top_n: int = 5


class ScreenRunResponse(BaseModel):
    run_id: int
    market_code: str
    universe_size: int
    filtered_size: int
    top_count: int
    bars_refreshed: int
    symbols_failed: list[str]
    llm_used: bool


class DataSourceOut(BaseModel):
    id: int
    name: str
    kind: str
    markets: list[str]
    config: dict[str, Any]
    priority: int
    enabled: bool
    last_health: str | None
    created_at: datetime


class DataSourceCreate(BaseModel):
    name: str
    kind: str
    markets: list[str] = Field(default_factory=list)
    config: dict[str, Any] = Field(default_factory=dict)
    priority: int = 0


class DataSourceUpdate(BaseModel):
    """Partial update -- every field is optional; only set fields are applied."""

    name: str | None = None
    kind: str | None = None
    markets: list[str] | None = None
    config: dict[str, Any] | None = None
    priority: int | None = None
    enabled: bool | None = None


class SourceTestResult(BaseModel):
    ok: bool
    detail: str
    checked_at: datetime


class PaperOrderOut(BaseModel):
    id: int
    pick_id: int | None
    symbol: str
    market: str
    side: str
    qty: float
    order_type: str
    status: str
    fill_price: float | None
    slippage_bps: float | None
    created_at: datetime
    filled_at: datetime | None


class PaperPositionOut(BaseModel):
    id: int
    symbol: str
    market: str
    qty: float
    avg_price: float
    opened_at: datetime
    closed_at: datetime | None
    realized_pnl: float | None


class PaperEquityPointOut(BaseModel):
    id: int
    date: datetime
    market: str
    cash: float
    positions_value: float
    total_pnl: float


class PaperPositionsResponse(BaseModel):
    positions: list[PaperPositionOut]


class PaperOrdersResponse(BaseModel):
    orders: list[PaperOrderOut]


class PaperEquityResponse(BaseModel):
    points: list[PaperEquityPointOut]


class PaperResetResponse(BaseModel):
    ok: bool
    detail: str


class PickOutcomeOut(BaseModel):
    pick_id: int
    symbol: str
    market: str
    strategy: str
    picked_at: date
    entry: float
    stop: float | None
    target: float | None
    days_held: int
    status: str
    return_pct: float | None
    max_favorable_pct: float
    max_adverse_pct: float


class OutcomesResponse(BaseModel):
    outcomes: list[PickOutcomeOut]
    summary: dict[str, Any]


class AttributionRowOut(BaseModel):
    symbol: str
    market: str
    strategy: str
    picked_at: date
    fill_price: float
    qty: float
    exit_price: float | None
    pnl: float | None
    pnl_pct: float | None
    llm_verdict: str | None
    status: str


class AttributionResponse(BaseModel):
    rows: list[AttributionRowOut]
    summary: dict[str, Any]
