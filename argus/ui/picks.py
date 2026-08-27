"""Picks page (`/picks`): run selector + full candidate table with expandable detail."""

from __future__ import annotations

from typing import Any

from nicegui import ui
from nicegui.events import ValueChangeEventArguments

from argus.db.models import DailyPick, ScreenRun
from argus.ui.layout import page_frame
from argus.ui.queries import picks_for_run, recent_runs

_NA = "-"


def _run_label(run: ScreenRun) -> str:
    return f"{run.market} — {run.run_ts:%Y-%m-%d %H:%M} UTC (run {run.id})"


def _pick_expansion(p: DailyPick) -> None:
    verdict = p.llm_verdict_json or {}
    title = f"{p.symbol} — {p.strategy} — score {p.score:.1f}"
    with ui.expansion(title).classes("w-full"):
        with ui.grid(columns=2).classes("gap-x-4"):
            ui.label("Market:")
            ui.label(p.market)
            ui.label("Stage:")
            ui.label(p.stage or _NA)
            ui.label("Entry:")
            ui.label(str(p.entry) if p.entry is not None else _NA)
            ui.label("Stop:")
            ui.label(str(p.stop) if p.stop is not None else _NA)
            ui.label("Target:")
            ui.label(str(p.target) if p.target is not None else _NA)

        ui.label("Reason").classes("font-bold mt-2")
        ui.label(p.reason or _NA)

        if verdict:
            ui.label("LLM Verdict").classes("font-bold mt-2")
            ui.label(f"{verdict.get('verdict', _NA)} (confidence {verdict.get('confidence', _NA)})")
            ui.label("Thesis").classes("font-bold mt-2")
            ui.label(str(verdict.get("thesis") or _NA))
            ui.label("Risks").classes("font-bold mt-2")
            ui.label(str(verdict.get("risks") or _NA))

            votes = verdict.get("votes") or []
            if votes:
                ui.label("Council Votes").classes("font-bold mt-2")
                for vote in votes:
                    if not isinstance(vote, dict):
                        continue
                    persona = vote.get("persona", _NA)
                    vote_verdict = vote.get("verdict", _NA)
                    confidence = vote.get("confidence", _NA)
                    thesis = str(vote.get("thesis") or _NA)
                    ui.label(f"{persona}: {vote_verdict} (confidence {confidence}) — {thesis}")

        if p.features_json:
            ui.label("Features").classes("font-bold mt-2")
            ui.markdown(
                "\n".join(f"- **{k}**: {v}" for k, v in sorted(p.features_json.items()))
            )


async def _render_picks(container: ui.column, run_id: int) -> None:
    container.clear()
    picks = await picks_for_run(run_id)
    with container:
        if not picks:
            ui.label("No picks for this run.").classes("text-gray-500")
            return
        for p in picks:
            _pick_expansion(p)


@ui.page("/picks")
async def picks_page() -> None:
    with page_frame("/picks"):
        ui.label("Picks").classes("text-2xl font-bold")
        runs = await recent_runs()
        content = ui.column().classes("w-full gap-2")

        if not runs:
            with content:
                ui.label("No screen runs yet.").classes("text-gray-500")
            return

        options = {run.id: _run_label(run) for run in runs}

        async def _on_change(e: ValueChangeEventArguments[Any]) -> None:
            await _render_picks(content, e.value)

        ui.select(options=options, value=runs[0].id, label="Run", on_change=_on_change).classes(
            "w-96"
        )
        await _render_picks(content, runs[0].id)
