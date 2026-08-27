"""Dashboard page (`/`): per-market status cards + today's top picks across markets."""

from __future__ import annotations

import asyncio
from typing import Any

import structlog
from nicegui import ui

from argus.db.models import DailyPick
from argus.markets import all_markets
from argus.pipeline import run_daily_pipeline
from argus.ui.layout import page_frame
from argus.ui.queries import latest_run_for_market, picks_for_run, top_picks_across_markets

logger = structlog.get_logger(__name__)

_NA = "-"


def _verdict_badge(llm_verdict_json: dict[str, Any] | None) -> str:
    if not llm_verdict_json:
        return _NA
    verdict = llm_verdict_json.get("verdict", _NA)
    confidence = llm_verdict_json.get("confidence", _NA)
    return f"{verdict} ({confidence})"


async def _run_market_screen(market_code: str, status_label: ui.label, spinner: ui.spinner) -> None:
    """Background task body for the "Run screen now" button.

    Run via ``asyncio.create_task`` (see ``_trigger_run`` below) so the click
    handler itself returns immediately and the event loop stays free to serve
    every other client while a run (which can take minutes with live data)
    is in flight.
    """
    spinner.visible = True
    status_label.text = "Running..."
    try:
        report = await run_daily_pipeline(market_code)
        status_label.text = f"Done: {len(report.result.top)} picks"
        ui.notify(f"{market_code} screen complete", type="positive")
    except Exception as exc:  # a UI action must never crash the page
        logger.error("ui.dashboard.run_failed", market=market_code, error=str(exc))
        status_label.text = "Failed"
        ui.notify(f"{market_code} screen failed: {exc}", type="negative")
    finally:
        spinner.visible = False


def _trigger_run(market_code: str, status_label: ui.label, spinner: ui.spinner) -> None:
    asyncio.create_task(_run_market_screen(market_code, status_label, spinner))


async def _market_card(market_code: str) -> None:
    run = await latest_run_for_market(market_code)
    with ui.card().classes("w-72"):
        ui.label(market_code).classes("text-lg font-bold")
        if run is None:
            ui.label("No runs yet").classes("text-sm text-gray-500")
        else:
            picks = await picks_for_run(run.id)
            ui.label(f"Last run: {run.run_ts:%Y-%m-%d %H:%M} UTC").classes("text-sm")
            ui.label(f"Universe: {run.universe_size} | Picks: {len(picks)}").classes("text-sm")
        status_label = ui.label("").classes("text-sm text-gray-500")
        with ui.row().classes("items-center gap-2"):
            spinner = ui.spinner(size="sm")
            spinner.visible = False
            ui.button(
                "Run screen now",
                on_click=lambda: _trigger_run(market_code, status_label, spinner),
            )


def _picks_table_rows(picks: list[DailyPick]) -> list[dict[str, Any]]:
    return [
        {
            "symbol": p.symbol,
            "market": p.market,
            "strategy": p.strategy,
            "score": round(p.score, 1),
            "entry": p.entry,
            "stop": p.stop,
            "target": p.target,
            "verdict": _verdict_badge(p.llm_verdict_json),
        }
        for p in picks
    ]


_PICKS_COLUMNS: list[dict[str, Any]] = [
    {"name": "symbol", "label": "Symbol", "field": "symbol", "sortable": True},
    {"name": "market", "label": "Market", "field": "market", "sortable": True},
    {"name": "strategy", "label": "Strategy", "field": "strategy"},
    {"name": "score", "label": "Score", "field": "score", "sortable": True},
    {"name": "entry", "label": "Entry", "field": "entry"},
    {"name": "stop", "label": "Stop", "field": "stop"},
    {"name": "target", "label": "Target", "field": "target"},
    {"name": "verdict", "label": "LLM Verdict", "field": "verdict"},
]


@ui.page("/")
async def dashboard_page() -> None:
    with page_frame("/"):
        ui.label("Today's Screen").classes("text-2xl font-bold")
        with ui.row().classes("gap-4 flex-wrap"):
            for market in all_markets():
                await _market_card(market.code)

        ui.label("Top Picks Across Markets").classes("text-xl font-bold mt-4")
        picks = await top_picks_across_markets()
        if picks:
            ui.table(
                columns=_PICKS_COLUMNS, rows=_picks_table_rows(picks), row_key="symbol"
            ).classes("w-full")
        else:
            ui.label("No picks yet today.").classes("text-gray-500")
