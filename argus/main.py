"""Entry point for the ARGUS application: FastAPI REST API + NiceGUI web UI.

``create_app()`` builds the combined app once (cached at module level, since
NiceGUI pages register onto a process-wide singleton and re-registering /
re-mounting them is unsupported); ``run()`` is the ``argus`` console-script
target that serves it with uvicorn.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
import uvicorn
from fastapi import FastAPI
from nicegui import ui

from argus.api.router import router as api_router
from argus.config import get_settings
from argus.data.sources import ensure_default_sources
from argus.db import init_db
from argus.jobs.scheduler import start_scheduler

logger = structlog.get_logger(__name__)

_app: FastAPI | None = None


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    await init_db(settings)
    await ensure_default_sources(settings)
    scheduler = start_scheduler(settings)
    try:
        yield
    finally:
        if settings.scheduler.enabled:
            scheduler.shutdown(wait=False)


def create_app() -> FastAPI:
    """Build (once) the FastAPI app: the REST API plus every NiceGUI page mounted on it."""
    global _app
    if _app is not None:
        return _app

    app = FastAPI(title="ARGUS", lifespan=_lifespan)
    app.include_router(api_router)

    import argus.ui.pages  # noqa: F401 -- registers @ui.page routes as a side effect

    ui.run_with(app, title="ARGUS", dark=True)

    _app = app
    return app


def run() -> None:
    """``argus`` console-script entry point: boot the combined API/UI server."""
    settings = get_settings()
    app = create_app()
    url = f"http://{settings.ui.host}:{settings.ui.port}"
    logger.info("argus.main.starting", url=url)
    uvicorn.run(app, host=settings.ui.host, port=settings.ui.port)
