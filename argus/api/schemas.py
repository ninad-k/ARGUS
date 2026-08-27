"""Pydantic request/response schemas for the REST API.

These are pure DTOs -- conversion to/from the SQLAlchemy models lives in the
router modules, not here.
"""

from __future__ import annotations

from datetime import datetime
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
