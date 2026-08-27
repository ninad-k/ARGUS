"""REST API tests: sources CRUD/test round-trip, picks/runs reads, screen/run.

The heavy ``POST /screen/run`` path (a real ``run_daily_pipeline`` call) is
never exercised here -- ``run_daily_pipeline`` is monkeypatched to return a
canned ``ScreenReport`` instead, matching how ``tests/test_scheduler_jobs.py``
avoids the network/DB-heavy pipeline.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from argus.api.router import router as api_router
from argus.config import AppSettings
from argus.db import async_session, init_db
from argus.db.models import DailyPick, ScreenRun
from argus.pipeline import ScreenReport
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
    monkeypatch.setattr("argus.api.picks.get_settings", lambda: settings)
    monkeypatch.setattr("argus.api.sources.get_settings", lambda: settings)
    await init_db(settings)

    app = _build_app()
    transport = ASGITransport(app=app)
    client = AsyncClient(transport=transport, base_url="http://test")
    return client, settings


async def test_create_list_patch_delete_source_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _ = await _client(tmp_path, monkeypatch)
    async with client:
        create_resp = await client.post(
            "/api/v1/sources",
            json={
                "name": "test-static",
                "kind": "static",
                "markets": ["US_NASDAQ"],
                "config": {},
                "priority": 1,
            },
        )
        assert create_resp.status_code == 201
        created = create_resp.json()
        assert created["name"] == "test-static"
        assert created["enabled"] is True
        source_id = created["id"]

        list_resp = await client.get("/api/v1/sources")
        assert list_resp.status_code == 200
        names = [s["name"] for s in list_resp.json()]
        assert "test-static" in names

        patch_resp = await client.patch(
            f"/api/v1/sources/{source_id}", json={"enabled": False, "priority": 5}
        )
        assert patch_resp.status_code == 200
        patched = patch_resp.json()
        assert patched["enabled"] is False
        assert patched["priority"] == 5

        delete_resp = await client.delete(f"/api/v1/sources/{source_id}")
        assert delete_resp.status_code == 204

        list_resp_after = await client.get("/api/v1/sources")
        names_after = [s["name"] for s in list_resp_after.json()]
        assert "test-static" not in names_after


async def test_patch_and_delete_unknown_source_return_404(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _ = await _client(tmp_path, monkeypatch)
    async with client:
        patch_resp = await client.patch("/api/v1/sources/9999", json={"enabled": False})
        assert patch_resp.status_code == 404

        delete_resp = await client.delete("/api/v1/sources/9999")
        assert delete_resp.status_code == 404

        test_resp = await client.post("/api/v1/sources/9999/test")
        assert test_resp.status_code == 404


async def test_source_test_endpoint_reports_static_provider_healthy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _ = await _client(tmp_path, monkeypatch)
    async with client:
        create_resp = await client.post(
            "/api/v1/sources",
            json={"name": "static-1", "kind": "static", "markets": [], "config": {}},
        )
        source_id = create_resp.json()["id"]

        test_resp = await client.post(f"/api/v1/sources/{source_id}/test")
        assert test_resp.status_code == 200
        body = test_resp.json()
        assert body["ok"] is True

        # health check result is persisted back onto the source row
        list_resp = await client.get("/api/v1/sources")
        source = next(s for s in list_resp.json() if s["id"] == source_id)
        assert source["last_health"] is not None


async def _insert_run_with_picks(settings: AppSettings, market: str, symbol: str) -> int:
    async with async_session(settings) as session:
        run = ScreenRun(
            market=market,
            run_ts=datetime.now(UTC),
            universe_size=10,
            strategies_json={"strategies": ["momentum"]},
            status="completed",
            duration_ms=123,
        )
        session.add(run)
        await session.flush()

        session.add(
            DailyPick(
                run_id=run.id,
                symbol=symbol,
                market=market,
                strategy="momentum",
                score=88.5,
                stage="breakout",
                reason="strong trend",
                entry=100.0,
                stop=95.0,
                target=115.0,
                features_json={"rsi_14": 65.0},
                llm_verdict_json={
                    "symbol": symbol,
                    "verdict": "buy",
                    "confidence": 80,
                    "thesis": "t",
                    "risks": "r",
                },
                created_at=run.run_ts,
            )
        )
        await session.commit()
        return run.id


async def test_picks_latest_returns_data_for_market(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, settings = await _client(tmp_path, monkeypatch)
    await _insert_run_with_picks(settings, "US_NASDAQ", "MOMO")

    async with client:
        resp = await client.get("/api/v1/picks/latest", params={"market": "US_NASDAQ"})
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["markets"]) == 1
        entry = body["markets"][0]
        assert entry["run"]["market"] == "US_NASDAQ"
        assert len(entry["picks"]) == 1
        pick = entry["picks"][0]
        assert pick["symbol"] == "MOMO"
        assert pick["llm_verdict_json"]["verdict"] == "buy"


async def test_picks_latest_without_market_covers_all_markets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, settings = await _client(tmp_path, monkeypatch)
    await _insert_run_with_picks(settings, "US_NASDAQ", "MOMO")
    await _insert_run_with_picks(settings, "IN_NSE", "RELI")

    async with client:
        resp = await client.get("/api/v1/picks/latest")
        assert resp.status_code == 200
        markets = {entry["run"]["market"] for entry in resp.json()["markets"]}
        assert markets == {"US_NASDAQ", "IN_NSE"}


async def test_picks_latest_unknown_market_returns_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _ = await _client(tmp_path, monkeypatch)
    async with client:
        resp = await client.get("/api/v1/picks/latest", params={"market": "US_NASDAQ"})
        assert resp.status_code == 200
        assert resp.json()["markets"] == []


async def test_runs_list_returns_recent_runs_newest_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, settings = await _client(tmp_path, monkeypatch)
    await _insert_run_with_picks(settings, "US_NASDAQ", "FIRST")
    await _insert_run_with_picks(settings, "US_NASDAQ", "SECOND")

    async with client:
        resp = await client.get("/api/v1/runs")
        assert resp.status_code == 200
        runs = resp.json()["runs"]
        assert len(runs) == 2
        assert runs[0]["id"] > runs[1]["id"]


async def test_screen_run_returns_summary_from_monkeypatched_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _ = await _client(tmp_path, monkeypatch)

    fake_result = ScreenResult(
        market_code="US_NASDAQ",
        run_ts=datetime.now(UTC),
        universe_size=42,
        filtered_size=10,
        candidates=[],
        top=[],
    )
    fake_report = ScreenReport(
        result=fake_result,
        run_id=7,
        bars_refreshed=3,
        symbols_failed=["BAD"],
        llm_used=True,
    )

    async def _fake_pipeline(market_code: str, *, top_n: int = 5, **kwargs: Any) -> ScreenReport:
        assert market_code == "US_NASDAQ"
        assert top_n == 3
        return fake_report

    monkeypatch.setattr("argus.api.picks.run_daily_pipeline", _fake_pipeline)

    async with client:
        resp = await client.post(
            "/api/v1/screen/run", json={"market_code": "US_NASDAQ", "top_n": 3}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["run_id"] == 7
        assert body["universe_size"] == 42
        assert body["bars_refreshed"] == 3
        assert body["symbols_failed"] == ["BAD"]
        assert body["llm_used"] is True


async def test_screen_run_unknown_market_returns_404(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _ = await _client(tmp_path, monkeypatch)

    async def _raise(market_code: str, **kwargs: Any) -> ScreenReport:
        raise KeyError(f"Unknown market code: {market_code!r}")

    monkeypatch.setattr("argus.api.picks.run_daily_pipeline", _raise)

    async with client:
        resp = await client.post("/api/v1/screen/run", json={"market_code": "NOPE"})
        assert resp.status_code == 404
