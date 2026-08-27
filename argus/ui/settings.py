"""Settings page (`/settings`): read-only config view + an Ollama connectivity check."""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
from nicegui import ui

from argus.config import AppSettings, get_settings
from argus.jobs.scheduler import build_scheduler
from argus.ui.layout import page_frame

_OLLAMA_PROBE_TIMEOUT_S = 2.0


async def _check_ollama(base_url: str, result_label: ui.label) -> None:
    result_label.text = "Checking..."
    try:
        async with httpx.AsyncClient(timeout=_OLLAMA_PROBE_TIMEOUT_S) as client:
            resp = await client.get(f"{base_url.rstrip('/')}/api/tags")
        if resp.status_code == 200:
            result_label.text = "Reachable"
        else:
            result_label.text = f"HTTP {resp.status_code}"
    except Exception as exc:  # a probe failure must never crash the page
        result_label.text = f"Unreachable: {exc}"


def _next_run_times(settings: AppSettings) -> list[tuple[str, str]]:
    """Job name -> next fire time, computed from an unstarted scheduler.

    Building a throwaway ``build_scheduler`` here (rather than reading the
    live scheduler's state) keeps this page independent of whatever process
    -- or none -- is actually running the scheduler.
    """
    scheduler = build_scheduler(settings)
    now = datetime.now(UTC)
    rows: list[tuple[str, str]] = []
    for job in scheduler.get_jobs():
        next_fire = job.trigger.get_next_fire_time(None, now)
        rows.append((job.name, next_fire.strftime("%Y-%m-%d %H:%M %Z") if next_fire else "-"))
    return rows


@ui.page("/settings")
async def settings_page() -> None:
    settings = get_settings()
    with page_frame("/settings"):
        ui.label("Settings").classes("text-2xl font-bold")

        with ui.card():
            ui.label("LLM").classes("text-lg font-bold")
            ui.label(f"Provider: {settings.llm.provider}")
            ui.label(f"Model: {settings.llm.model}")
            ui.label(f"Base URL: {settings.llm.base_url}")
            ui.label(f"Enabled: {settings.llm.enabled}")
            result_label = ui.label("").classes("text-sm")
            ui.button(
                "Check Ollama connectivity",
                on_click=lambda: _check_ollama(settings.llm.base_url, result_label),
            )

        with ui.card():
            ui.label("Data").classes("text-lg font-bold")
            ui.label(f"Data dir: {settings.data_dir}")

        with ui.card():
            ui.label("Scheduler").classes("text-lg font-bold")
            ui.label(f"Enabled: {settings.scheduler.enabled}")
            if settings.scheduler.enabled:
                for name, next_run in _next_run_times(settings):
                    ui.label(f"{name}: next run {next_run}")
            else:
                ui.label("Scheduler disabled.")
