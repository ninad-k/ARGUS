"""TradingView alert webhook receiver.

To configure a TradingView alert to post here: in the alert dialog's
"Webhook URL" field, use
``https://<your-argus-host>/api/v1/webhooks/tradingview/<token>``, where
``<token>`` is ``ARGUS_WEBHOOKS_TRADINGVIEW_TOKEN`` (see
``argus.config.webhooks.WebhookSettings``). TradingView alert messages may be
freeform JSON (e.g. a custom message body referencing ``{{ticker}}``,
``{{close}}``, ``{{strategy.order.action}}``, ...) or plain text -- both are
accepted; plain text is stored wrapped as ``{"text": "..."}``.

An empty/unset token disables the endpoint entirely (every request 404s,
whatever token it carries) -- ARGUS ships with no webhook receiver active by
default.

Phase 2 scope is intentionally minimal: store the event and expose a read
endpoint for the UI. Turning a webhook event into a screener signal /
``DailyPick`` is a later phase -- see the task description for Task 9.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import select

from argus.config import get_settings
from argus.db import async_session
from argus.db.models import WebhookEvent

router = APIRouter(prefix="/api/v1/webhooks", tags=["webhooks"])


class WebhookEventOut(BaseModel):
    id: int
    source: str
    payload: dict[str, Any]
    received_at: datetime
    processed: bool


@router.post("/tradingview/{token}", status_code=201)
async def receive_tradingview_webhook(token: str, request: Request) -> dict[str, Any]:
    settings = get_settings()
    expected = settings.webhooks.tradingview_token
    if not expected or token != expected:
        # Don't distinguish "disabled" from "wrong token" in the response --
        # both look identical to an outside caller.
        raise HTTPException(status_code=404, detail="not_found")

    payload = await _parse_body(request)

    async with async_session(settings) as session:
        event = WebhookEvent(
            source="tradingview",
            payload_json=payload,
            received_at=datetime.now(UTC),
            processed=False,
        )
        session.add(event)
        await session.commit()
        await session.refresh(event)
        event_id = event.id

    return {"status": "ok", "id": event_id}


@router.get("/events", response_model=list[WebhookEventOut])
async def list_webhook_events(limit: int = 50) -> list[WebhookEventOut]:
    settings = get_settings()
    async with async_session(settings) as session:
        result = await session.execute(
            select(WebhookEvent).order_by(WebhookEvent.received_at.desc()).limit(limit)
        )
        rows = result.scalars().all()
    return [
        WebhookEventOut(
            id=row.id,
            source=row.source,
            payload=row.payload_json,
            received_at=row.received_at,
            processed=row.processed,
        )
        for row in rows
    ]


async def _parse_body(request: Request) -> dict[str, Any]:
    """TradingView alerts may post JSON or plain text -- accept both."""
    raw = await request.body()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except ValueError:
        return {"text": raw.decode("utf-8", errors="replace")}
    if isinstance(parsed, dict):
        return parsed
    return {"value": parsed}
