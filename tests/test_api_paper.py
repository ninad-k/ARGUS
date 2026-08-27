"""``/api/v1/paper`` endpoints: positions/orders/equity reads round-trip
against fixtures, and reset wipes everything and re-seeds cash."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from argus.api.router import router as api_router
from argus.config import AppSettings
from argus.db import async_session, init_db
from argus.db.models import PaperCash, PaperEquityPoint, PaperOrder, PaperPosition
from argus.paper.portfolio import US_DOMAIN, ensure_cash_initialized


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
    monkeypatch.setattr("argus.api.paper.get_settings", lambda: settings)
    await init_db(settings)

    app = _build_app()
    transport = ASGITransport(app=app)
    client = AsyncClient(transport=transport, base_url="http://test")
    return client, settings


async def _seed_fixtures(settings: AppSettings) -> None:
    now = datetime.now(UTC)
    await ensure_cash_initialized(US_DOMAIN, settings)
    async with async_session(settings) as session:
        session.add(
            PaperPosition(
                symbol="AAPL", market="US_NASDAQ", qty=10.0, avg_price=100.0, opened_at=now
            )
        )
        session.add(
            PaperOrder(
                symbol="AAPL",
                market="US_NASDAQ",
                side="buy",
                qty=10.0,
                order_type="next_open",
                status="filled",
                fill_price=100.0,
                slippage_bps=5.0,
                created_at=now,
                filled_at=now,
            )
        )
        session.add(
            PaperEquityPoint(
                date=now, market="US_NASDAQ", cash=99_000.0, positions_value=1_000.0, total_pnl=0.0
            )
        )
        await session.commit()


async def test_get_positions_round_trips_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, settings = await _client(tmp_path, monkeypatch)
    await _seed_fixtures(settings)

    async with client:
        resp = await client.get("/api/v1/paper/positions")
        assert resp.status_code == 200
        positions = resp.json()["positions"]
        assert len(positions) == 1
        assert positions[0]["symbol"] == "AAPL"
        assert positions[0]["qty"] == 10.0


async def test_get_orders_filters_by_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, settings = await _client(tmp_path, monkeypatch)
    await _seed_fixtures(settings)
    now = datetime.now(UTC)
    async with async_session(settings) as session:
        session.add(
            PaperOrder(
                symbol="MSFT",
                market="US_NASDAQ",
                side="buy",
                qty=5.0,
                order_type="next_open",
                status="pending",
                created_at=now,
            )
        )
        await session.commit()

    async with client:
        resp_all = await client.get("/api/v1/paper/orders")
        assert len(resp_all.json()["orders"]) == 2

        resp_pending = await client.get("/api/v1/paper/orders", params={"status": "pending"})
        pending = resp_pending.json()["orders"]
        assert len(pending) == 1
        assert pending[0]["symbol"] == "MSFT"


async def test_get_equity_filters_by_market(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, settings = await _client(tmp_path, monkeypatch)
    await _seed_fixtures(settings)
    now = datetime.now(UTC)
    async with async_session(settings) as session:
        session.add(
            PaperEquityPoint(
                date=now, market="IN_NSE", cash=500_000.0, positions_value=0.0, total_pnl=0.0
            )
        )
        await session.commit()

    async with client:
        resp = await client.get("/api/v1/paper/equity", params={"market": "US_NASDAQ"})
        points = resp.json()["points"]
        assert len(points) == 1
        assert points[0]["market"] == "US_NASDAQ"

        resp_all = await client.get("/api/v1/paper/equity")
        assert len(resp_all.json()["points"]) == 2


async def test_reset_clears_everything_and_reseeds_cash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, settings = await _client(tmp_path, monkeypatch)
    await _seed_fixtures(settings)

    async with client:
        resp = await client.post("/api/v1/paper/reset")
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

        positions_resp = await client.get("/api/v1/paper/positions")
        assert positions_resp.json()["positions"] == []

        orders_resp = await client.get("/api/v1/paper/orders")
        assert orders_resp.json()["orders"] == []

        equity_resp = await client.get("/api/v1/paper/equity")
        assert equity_resp.json()["points"] == []

    async with async_session(settings) as session:
        cash_rows = (await session.execute(select(PaperCash))).scalars().all()
        assert {r.domain for r in cash_rows} == {"US", "IN"}
        us_row = next(r for r in cash_rows if r.domain == "US")
        assert us_row.cash == settings.paper.starting_cash_us
