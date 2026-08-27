"""Aggregates the versioned REST API routers under one ``APIRouter``."""

from fastapi import APIRouter

from argus.api.paper import router as paper_router
from argus.api.picks import router as picks_router
from argus.api.sources import router as sources_router
from argus.api.webhooks import router as webhooks_router

router = APIRouter()
router.include_router(picks_router)
router.include_router(sources_router)
router.include_router(paper_router)
router.include_router(webhooks_router)

__all__ = ["router"]
