"""Import-level smoke tests: building the app registers every route without error.

Full browser-driven UI tests are out of scope for Phase 1 -- this only
verifies that ``create_app()`` succeeds and that the REST API routes and the
four NiceGUI ``@ui.page`` routes all end up registered.
"""

from __future__ import annotations

from nicegui import Client

from argus.main import create_app


def test_create_app_registers_api_routes() -> None:
    # FastAPI's route tree doesn't expose included-router paths as flat
    # `route.path` attributes on newer versions -- the OpenAPI schema is the
    # version-stable way to check which paths actually got registered.
    app = create_app()
    paths = set(app.openapi()["paths"])
    assert "/api/v1/sources" in paths
    assert "/api/v1/picks/latest" in paths
    assert "/api/v1/runs" in paths
    assert "/api/v1/paper/positions" in paths
    assert "/api/v1/paper/orders" in paths
    assert "/api/v1/paper/equity" in paths
    assert "/api/v1/paper/reset" in paths


def test_create_app_registers_nicegui_pages() -> None:
    create_app()
    page_paths = set(Client.page_routes.values())
    assert {"/", "/picks", "/paper", "/sources", "/settings"} <= page_paths


def test_create_app_is_cached_across_calls() -> None:
    first = create_app()
    second = create_app()
    assert first is second
