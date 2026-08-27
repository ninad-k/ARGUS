"""``GET /api/v1/reports/latest`` -- the newest saved Markdown daily report.

A simple file lookup (see ``argus.reports.latest_report_path``), not a DB
read -- reports are written to disk by the scheduler/smoke script, this just
serves the newest one back as plain text.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse

from argus.config import get_settings
from argus.reports import latest_report_path

router = APIRouter(prefix="/api/v1/reports", tags=["reports"])


@router.get("/latest", response_class=PlainTextResponse)
async def get_latest_report(market: str | None = Query(default=None)) -> PlainTextResponse:
    settings = get_settings()
    path = latest_report_path(market, settings=settings, fmt="md")
    if path is None:
        raise HTTPException(status_code=404, detail="No saved report found")
    return PlainTextResponse(path.read_text(encoding="utf-8"))
