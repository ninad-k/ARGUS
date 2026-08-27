"""Sources page (`/sources`): the data-source admin screen -- the "multiple
sources" UI called for by the product requirements.

All CRUD/health-check calls go straight through ``argus.data.sources`` (the
same functions the REST API uses), never through an HTTP call to the API --
see ``argus.ui.queries`` module docstring.
"""

from __future__ import annotations

from typing import Any

from nicegui import ui
from nicegui.events import ValueChangeEventArguments

from argus.data.sources import (
    check_source_health,
    create_source,
    delete_source,
    list_sources,
    update_source,
)
from argus.db.models import DataSource
from argus.ui.layout import page_frame

_KIND_OPTIONS: list[str] = ["yfinance", "tvscreener", "nse", "static"]
_MARKET_OPTIONS: list[str] = ["US_NYSE", "US_NASDAQ", "IN_NSE"]

# Shown next to each source row so an admin can tell at a glance what a kind
# is actually good for -- e.g. "nse" only ever serves IN_NSE quotes, and
# "tvscreener" has no OHLCV history at all (see the provider docstrings).
_KIND_CAPABILITIES: dict[str, str] = {
    "yfinance": "prices, quotes, fundamentals (all markets)",
    "tvscreener": "quotes, universe ranking, fundamentals (no OHLCV history)",
    "nse": "quotes only (IN_NSE only, no OHLCV history)",
    "static": "prices, quotes (test/fixture data)",
}


async def _refresh(container: ui.column) -> None:
    container.clear()
    sources = await list_sources()
    with container:
        if not sources:
            ui.label("No sources configured.").classes("text-gray-500")
        for source in sources:
            _source_row(source, container)


async def _toggle_enabled(source_id: int, enabled: bool, container: ui.column) -> None:
    await update_source(source_id, enabled=enabled)
    ui.notify("Source updated", type="info")
    await _refresh(container)


async def _delete(source_id: int, container: ui.column) -> None:
    await delete_source(source_id)
    ui.notify("Source deleted", type="info")
    await _refresh(container)


async def _test(source_id: int, container: ui.column) -> None:
    sources = await list_sources()
    source = next((s for s in sources if s.id == source_id), None)
    if source is None:
        return
    health = await check_source_health(source)
    ui.notify(
        f"{source.name}: {'OK' if health.ok else 'UNHEALTHY'} — {health.detail}",
        type="positive" if health.ok else "negative",
    )
    await _refresh(container)


def _source_row(source: DataSource, container: ui.column) -> None:
    source_id = source.id
    with ui.row().classes("w-full items-center gap-4 border-b py-2"):
        ui.label(source.name).classes("w-40")
        ui.label(source.kind).classes("w-24")
        ui.label(", ".join(source.markets_json.get("markets", []))).classes("w-48")
        ui.label(str(source.priority)).classes("w-16")
        ui.label(_KIND_CAPABILITIES.get(source.kind, "")).classes("w-64 text-xs text-gray-400")

        async def _on_toggle(e: ValueChangeEventArguments[Any], sid: int = source_id) -> None:
            await _toggle_enabled(sid, bool(e.value), container)

        ui.switch(value=source.enabled, on_change=_on_toggle)
        ui.label(source.last_health or "-").classes("flex-1 text-sm text-gray-500")
        ui.button("Test", on_click=lambda sid=source_id: _test(sid, container))
        ui.button("Delete", on_click=lambda sid=source_id: _delete(sid, container), color="red")


def _add_dialog(container: ui.column) -> ui.dialog:
    with ui.dialog() as dialog, ui.card():
        ui.label("Add Data Source").classes("text-lg font-bold")
        name_input = ui.input("Name")
        kind_select = ui.select(_KIND_OPTIONS, value=_KIND_OPTIONS[0], label="Kind")
        markets_select = ui.select(_MARKET_OPTIONS, multiple=True, label="Markets").classes("w-64")
        priority_input = ui.number("Priority", value=0, precision=0)

        async def _submit() -> None:
            name = name_input.value
            if not name:
                ui.notify("Name is required", type="warning")
                return
            await create_source(
                name=name,
                kind=kind_select.value,
                markets=list(markets_select.value or []),
                config={},
                priority=int(priority_input.value or 0),
            )
            dialog.close()
            ui.notify("Source added", type="positive")
            await _refresh(container)

        with ui.row():
            ui.button("Cancel", on_click=dialog.close)
            ui.button("Add", on_click=_submit)
    return dialog


@ui.page("/sources")
async def sources_page() -> None:
    with page_frame("/sources"):
        ui.label("Data Sources").classes("text-2xl font-bold")
        container = ui.column().classes("w-full")
        dialog = _add_dialog(container)
        ui.button("Add source", on_click=dialog.open)
        await _refresh(container)
