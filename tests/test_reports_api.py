"""``GET /api/v1/reports/latest`` -- 404 when nothing's saved, else the
newest saved Markdown report's raw text."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from argus.api.router import router as api_router
from argus.config import AppSettings
from argus.markets import US_NASDAQ
from argus.pipeline import ScreenReport
from argus.reports import save_report
from argus.screener.runner import ScreenResult


def _settings(tmp_path: Path) -> AppSettings:
    return AppSettings(data_dir=tmp_path, _env_file=None)  # type: ignore[call-arg]


def _build_app() -> FastAPI:
    app = FastAPI()
    app.include_router(api_router)
    return app


async def _client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[AsyncClient, AppSettings]:
    settings = _settings(tmp_path)
    monkeypatch.setattr("argus.api.reports.get_settings", lambda: settings)
    app = _build_app()
    transport = ASGITransport(app=app)
    client = AsyncClient(transport=transport, base_url="http://test")
    return client, settings


def _report(market_code: str = US_NASDAQ.code) -> ScreenReport:
    result = ScreenResult(
        market_code=market_code,
        run_ts=datetime(2026, 8, 27, 16, 30, 0, tzinfo=UTC),
        universe_size=1,
        filtered_size=1,
        candidates=[],
        top=[],
    )
    return ScreenReport(
        result=result, run_id=1, bars_refreshed=0, symbols_failed=[], llm_used=False
    )


async def test_get_latest_report_returns_404_when_nothing_saved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _ = await _client(tmp_path, monkeypatch)
    async with client:
        resp = await client.get("/api/v1/reports/latest", params={"market": US_NASDAQ.code})
    assert resp.status_code == 404


async def test_get_latest_report_returns_saved_markdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, settings = await _client(tmp_path, monkeypatch)
    save_report(_report(), out_dir=settings.data_dir / "reports")

    async with client:
        resp = await client.get("/api/v1/reports/latest", params={"market": US_NASDAQ.code})

    assert resp.status_code == 200
    assert "ARGUS Daily Picks" in resp.text
    assert resp.headers["content-type"].startswith("text/plain")
