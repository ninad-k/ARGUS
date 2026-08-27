"""``/api/v1/history`` endpoints against hand-seeded fixtures: outcomes
(bars + a pick) and attribution (a filled buy order + its pick)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from argus.api.router import router as api_router
from argus.config import AppSettings
from argus.data.prices.base import bars_from_columns
from argus.data.store.duckdb_ohlcv import BarStore
from argus.db import async_session, init_db
from argus.db.models import DailyPick, PaperOrder, PaperPosition, ScreenRun

_MARKET = "US_NASDAQ"


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
    monkeypatch.setattr("argus.api.history.get_settings", lambda: settings)
    monkeypatch.setattr("argus.analysis.attribution.get_settings", lambda: settings)
    await init_db(settings)

    app = _build_app()
    transport = ASGITransport(app=app)
    client = AsyncClient(transport=transport, base_url="http://test")
    return client, settings


def _put_bar(
    store: BarStore, symbol: str, day: datetime, high: float, low: float, close: float
) -> None:
    ts = np.array([np.datetime64(day.date().isoformat(), "s")], dtype="datetime64[s]")
    bars = bars_from_columns(
        ts,
        np.array([close]),
        np.array([high]),
        np.array([low]),
        np.array([close]),
        np.array([1_000_000.0]),
    )
    store.upsert_bars(_MARKET, symbol, bars)


async def _seed_outcome_pick(settings: AppSettings, tmp_path: Path) -> None:
    picked_at = datetime.now(UTC) - timedelta(days=2)
    async with async_session(settings) as session:
        run = ScreenRun(
            market=_MARKET,
            run_ts=picked_at,
            universe_size=1,
            strategies_json={},
            status="completed",
        )
        session.add(run)
        await session.flush()
        session.add(
            DailyPick(
                run_id=run.id,
                symbol="AAPL",
                market=_MARKET,
                strategy="momentum",
                score=90.0,
                stage="breakout",
                entry=100.0,
                stop=90.0,
                target=110.0,
                features_json={},
                created_at=picked_at,
            )
        )
        await session.commit()

    with BarStore(tmp_path / "market_data.duckdb") as store:
        _put_bar(store, "AAPL", picked_at + timedelta(days=1), high=112.0, low=99.0, close=111.0)


async def _seed_attribution_position(settings: AppSettings) -> None:
    now = datetime.now(UTC)
    async with async_session(settings) as session:
        pick = DailyPick(
            run_id=1,
            symbol="MSFT",
            market=_MARKET,
            strategy="breakout",
            score=88.0,
            stage="breakout",
            entry=200.0,
            stop=190.0,
            target=220.0,
            features_json={},
            llm_verdict_json={
                "symbol": "MSFT",
                "verdict": "buy",
                "confidence": 75,
                "thesis": "t",
                "risks": "r",
            },
            created_at=now,
        )
        session.add(pick)
        await session.flush()

        session.add(
            PaperOrder(
                pick_id=pick.id,
                symbol="MSFT",
                market=_MARKET,
                side="buy",
                qty=5.0,
                order_type="next_open",
                status="filled",
                fill_price=200.0,
                slippage_bps=0.0,
                created_at=now - timedelta(days=1),
                filled_at=now,
            )
        )
        session.add(
            PaperOrder(
                pick_id=pick.id,
                symbol="MSFT",
                market=_MARKET,
                side="sell",
                qty=5.0,
                order_type="next_open",
                status="filled",
                fill_price=220.0,
                slippage_bps=0.0,
                created_at=now,
                filled_at=now + timedelta(hours=1),
            )
        )
        session.add(
            PaperPosition(
                symbol="MSFT",
                market=_MARKET,
                qty=0.0,
                avg_price=200.0,
                opened_at=now,
                closed_at=now + timedelta(hours=1),
                realized_pnl=100.0,
            )
        )
        await session.commit()


async def test_get_outcomes_returns_evaluated_picks_and_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, settings = await _client(tmp_path, monkeypatch)
    await _seed_outcome_pick(settings, tmp_path)

    async with client:
        resp = await client.get("/api/v1/history/outcomes", params={"market": _MARKET})

    assert resp.status_code == 200
    body = resp.json()
    assert len(body["outcomes"]) == 1
    outcome = body["outcomes"][0]
    assert outcome["symbol"] == "AAPL"
    assert outcome["status"] == "hit_target"
    assert body["summary"]["hit_rate"] == pytest.approx(1.0)
    assert body["summary"]["counts"]["hit_target"] == 1
    assert "momentum" in body["summary"]["by_strategy"]


async def test_get_outcomes_empty_when_no_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _ = await _client(tmp_path, monkeypatch)

    async with client:
        resp = await client.get("/api/v1/history/outcomes")

    assert resp.status_code == 200
    body = resp.json()
    assert body["outcomes"] == []
    assert body["summary"]["total"] == 0


async def test_get_attribution_returns_rows_and_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, settings = await _client(tmp_path, monkeypatch)
    await _seed_attribution_position(settings)

    async with client:
        resp = await client.get("/api/v1/history/attribution")

    assert resp.status_code == 200
    body = resp.json()
    assert len(body["rows"]) == 1
    row = body["rows"][0]
    assert row["symbol"] == "MSFT"
    assert row["pnl"] == pytest.approx(100.0)
    assert row["llm_verdict"] == "buy"
    assert body["summary"]["total_pnl"] == pytest.approx(100.0)
    assert body["summary"]["by_verdict"]["buy"] == pytest.approx(100.0)


async def test_get_attribution_empty_when_no_orders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _ = await _client(tmp_path, monkeypatch)

    async with client:
        resp = await client.get("/api/v1/history/attribution")

    assert resp.status_code == 200
    body = resp.json()
    assert body["rows"] == []
    assert body["summary"]["total_pnl"] == 0.0
