"""TradingView webhook receiver: token gating (valid/wrong/disabled), JSON and
plain-text body storage, and the events read endpoint."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from argus.api.router import router as api_router
from argus.config import AppSettings
from argus.config.webhooks import WebhookSettings
from argus.db import init_db


def _settings(tmp_path: Path, *, token: str = "") -> AppSettings:
    return AppSettings(
        data_dir=tmp_path,
        webhooks=WebhookSettings(tradingview_token=token, _env_file=None),  # type: ignore[call-arg]
        _env_file=None,  # type: ignore[call-arg]
    )


def _build_app() -> FastAPI:
    app = FastAPI()
    app.include_router(api_router)
    return app


async def _client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, token: str = ""
) -> AsyncClient:
    settings = _settings(tmp_path, token=token)
    monkeypatch.setattr("argus.api.webhooks.get_settings", lambda: settings)
    await init_db(settings)

    app = _build_app()
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


async def test_valid_token_stores_json_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = await _client(tmp_path, monkeypatch, token="secret123")
    async with client:
        resp = await client.post(
            "/api/v1/webhooks/tradingview/secret123",
            json={"symbol": "AAPL", "action": "buy"},
        )
        assert resp.status_code == 201
        assert resp.json()["status"] == "ok"

        events = (await client.get("/api/v1/webhooks/events")).json()
        assert len(events) == 1
        assert events[0]["source"] == "tradingview"
        assert events[0]["payload"] == {"symbol": "AAPL", "action": "buy"}
        assert events[0]["processed"] is False


async def test_valid_token_stores_plain_text_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = await _client(tmp_path, monkeypatch, token="secret123")
    async with client:
        resp = await client.post(
            "/api/v1/webhooks/tradingview/secret123",
            content=b"AAPL crossed above 200 EMA",
            headers={"content-type": "text/plain"},
        )
        assert resp.status_code == 201

        events = (await client.get("/api/v1/webhooks/events")).json()
        assert events[0]["payload"] == {"text": "AAPL crossed above 200 EMA"}


async def test_wrong_token_returns_404_and_stores_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = await _client(tmp_path, monkeypatch, token="secret123")
    async with client:
        resp = await client.post("/api/v1/webhooks/tradingview/wrong-token", json={"a": 1})
        assert resp.status_code == 404

        events = (await client.get("/api/v1/webhooks/events")).json()
        assert events == []


async def test_disabled_when_no_token_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = await _client(tmp_path, monkeypatch, token="")
    async with client:
        resp = await client.post("/api/v1/webhooks/tradingview/anything", json={"a": 1})
        assert resp.status_code == 404


async def test_events_list_respects_limit_and_orders_newest_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = await _client(tmp_path, monkeypatch, token="tok")
    async with client:
        for i in range(3):
            await client.post("/api/v1/webhooks/tradingview/tok", json={"i": i})

        events = (await client.get("/api/v1/webhooks/events", params={"limit": 2})).json()

        assert len(events) == 2
        assert events[0]["payload"]["i"] == 2  # newest first
        assert events[1]["payload"]["i"] == 1


async def test_non_dict_json_body_is_wrapped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = await _client(tmp_path, monkeypatch, token="tok")
    async with client:
        resp = await client.post(
            "/api/v1/webhooks/tradingview/tok",
            content=b"[1, 2, 3]",
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 201

        events = (await client.get("/api/v1/webhooks/events")).json()
        assert events[0]["payload"] == {"value": [1, 2, 3]}


async def test_empty_body_stores_empty_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = await _client(tmp_path, monkeypatch, token="tok")
    async with client:
        resp = await client.post("/api/v1/webhooks/tradingview/tok", content=b"")
        assert resp.status_code == 201

        events = (await client.get("/api/v1/webhooks/events")).json()
        assert events[0]["payload"] == {}
