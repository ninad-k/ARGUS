"""Shared page chrome: a header nav bar linking the four pages."""

from __future__ import annotations

from nicegui import ui

_PAGES: tuple[tuple[str, str], ...] = (
    ("Dashboard", "/"),
    ("Picks", "/picks"),
    ("Paper", "/paper"),
    ("Sources", "/sources"),
    ("Settings", "/settings"),
)


def header(active: str) -> None:
    """Render the shared top nav bar. ``active`` is the current page's path."""
    with ui.header().classes("items-center justify-between"):
        ui.label("ARGUS").classes("text-xl font-bold")
        with ui.row().classes("gap-4"):
            for label, path in _PAGES:
                link = ui.link(label, path).classes("text-white no-underline")
                if path == active:
                    link.classes("font-bold underline")


def page_frame(active: str) -> ui.column:
    """Header + a padded content column. Use as ``with page_frame(...):``."""
    header(active)
    return ui.column().classes("w-full p-4 gap-4")
