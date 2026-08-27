"""Aggregates the versioned REST API routers under one ``APIRouter``.

Webhooks (ingesting third-party events -- TradingView alerts, broker
notifications) are a Phase 2 concern; no webhook router exists yet.
"""

from fastapi import APIRouter

from argus.api.picks import router as picks_router
from argus.api.sources import router as sources_router

router = APIRouter()
router.include_router(picks_router)
router.include_router(sources_router)

__all__ = ["router"]
