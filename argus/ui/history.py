"""History page (`/history`): pick-outcome review + paper-vs-pick attribution.

Answers two questions with real numbers instead of vibes: "did our picks
actually hit their targets" (``argus.analysis.outcomes``) and "which
strategies/LLM verdicts made the (simulated) paper account money"
(``argus.analysis.attribution``). Both are read-only analytics -- nothing on
this page mutates a pick or an order.
"""

from __future__ import annotations

from typing import Any

from nicegui import ui

from argus.analysis.attribution import attribution_summary, paper_attribution
from argus.analysis.outcomes import PickOutcome, evaluate_run_history, summarize_outcomes
from argus.config import get_settings
from argus.data.store.duckdb_ohlcv import BarStore
from argus.ui.layout import page_frame

_NA = "-"
_LIMIT_RUNS = 30

_OUTCOME_COLUMNS: list[dict[str, Any]] = [
    {"name": "symbol", "label": "Symbol", "field": "symbol", "sortable": True},
    {"name": "market", "label": "Market", "field": "market"},
    {"name": "strategy", "label": "Strategy", "field": "strategy"},
    {"name": "picked_at", "label": "Picked", "field": "picked_at", "sortable": True},
    {"name": "status", "label": "Status", "field": "status"},
    {"name": "return_pct", "label": "Return %", "field": "return_pct", "sortable": True},
    {"name": "mfe", "label": "MFE %", "field": "mfe"},
    {"name": "mae", "label": "MAE %", "field": "mae"},
    {"name": "days_held", "label": "Days Held", "field": "days_held"},
]

_ATTRIBUTION_COLUMNS: list[dict[str, Any]] = [
    {"name": "symbol", "label": "Symbol", "field": "symbol", "sortable": True},
    {"name": "market", "label": "Market", "field": "market"},
    {"name": "strategy", "label": "Strategy", "field": "strategy"},
    {"name": "picked_at", "label": "Picked", "field": "picked_at"},
    {"name": "fill_price", "label": "Fill", "field": "fill_price"},
    {"name": "exit_price", "label": "Exit", "field": "exit_price"},
    {"name": "pnl", "label": "P&L", "field": "pnl", "sortable": True},
    {"name": "pnl_pct", "label": "P&L %", "field": "pnl_pct"},
    {"name": "llm_verdict", "label": "LLM Verdict", "field": "llm_verdict"},
    {"name": "status", "label": "Status", "field": "status"},
]

_STRATEGY_STATS_COLUMNS: list[dict[str, Any]] = [
    {"name": "strategy", "label": "Strategy", "field": "strategy"},
    {"name": "total", "label": "Picks", "field": "total"},
    {"name": "hit_rate", "label": "Hit Rate", "field": "hit_rate"},
    {"name": "stop_rate", "label": "Stop Rate", "field": "stop_rate"},
    {"name": "avg_return_pct", "label": "Avg Return %", "field": "avg_return_pct"},
    {"name": "expectancy", "label": "Expectancy", "field": "expectancy"},
]


def _outcome_rows(outcomes: list[PickOutcome]) -> list[dict[str, Any]]:
    return [
        {
            "symbol": o.symbol,
            "market": o.market,
            "strategy": o.strategy,
            "picked_at": o.picked_at.isoformat(),
            "status": o.status,
            "return_pct": o.return_pct if o.return_pct is not None else _NA,
            "mfe": o.max_favorable_pct,
            "mae": o.max_adverse_pct,
            "days_held": o.days_held,
        }
        for o in outcomes
    ]


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _strategy_stats_rows(by_strategy: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "strategy": strategy,
            "total": stats["total"],
            "hit_rate": _pct(stats["hit_rate"]),
            "stop_rate": _pct(stats["stop_rate"]),
            "avg_return_pct": stats["avg_return_pct"],
            "expectancy": stats["expectancy"],
        }
        for strategy, stats in sorted(by_strategy.items())
    ]


async def _render_outcomes(summary: dict[str, Any], outcomes: list[PickOutcome]) -> None:
    with ui.row().classes("gap-4 flex-wrap"):
        with ui.card().classes("w-48"):
            ui.label("Hit Rate").classes("text-sm text-gray-500")
            ui.label(_pct(summary["hit_rate"])).classes("text-xl font-bold")
        with ui.card().classes("w-48"):
            ui.label("Stop Rate").classes("text-sm text-gray-500")
            ui.label(_pct(summary["stop_rate"])).classes("text-xl font-bold")
        with ui.card().classes("w-48"):
            ui.label("Avg Return %").classes("text-sm text-gray-500")
            ui.label(f"{summary['avg_return_pct']:.2f}%").classes("text-xl font-bold")
        with ui.card().classes("w-48"):
            ui.label("Avg Winner %").classes("text-sm text-gray-500")
            ui.label(f"{summary['avg_winner_pct']:.2f}%").classes("text-xl font-bold")
        with ui.card().classes("w-48"):
            ui.label("Avg Loser %").classes("text-sm text-gray-500")
            ui.label(f"{summary['avg_loser_pct']:.2f}%").classes("text-xl font-bold")
        with ui.card().classes("w-48"):
            ui.label("Expectancy").classes("text-sm text-gray-500")
            ui.label(f"{summary['expectancy']:.2f}%").classes("text-xl font-bold")

    ui.label("By Strategy").classes("text-lg font-bold mt-4")
    strategy_rows = _strategy_stats_rows(summary["by_strategy"])
    if strategy_rows:
        ui.table(
            columns=_STRATEGY_STATS_COLUMNS, rows=strategy_rows, row_key="strategy"
        ).classes("w-full")
    else:
        ui.label("No evaluated picks yet.").classes("text-gray-500")

    ui.label("Pick Outcomes").classes("text-lg font-bold mt-4")
    if not outcomes:
        ui.label("No evaluated picks yet -- picks need at least one bar after their "
                  "pick date to be scored.").classes("text-gray-500")
        return

    table = ui.table(
        columns=_OUTCOME_COLUMNS, rows=_outcome_rows(outcomes), row_key="symbol"
    ).classes("w-full")
    table.add_slot(
        "body-cell-status",
        r'''
        <q-td :props="props">
            <q-badge :color="props.value === 'hit_target' ? 'green'
                : props.value === 'hit_stop' ? 'red'
                : props.value === 'open' ? 'blue' : 'grey'">
                {{ props.value }}
            </q-badge>
        </q-td>
        ''',
    )


def _attribution_mini_table(title: str, by_key: dict[str, float]) -> None:
    ui.label(title).classes("font-bold mt-2")
    if not by_key:
        ui.label("No priced positions yet.").classes("text-gray-500 text-sm")
        return
    with ui.grid(columns=2).classes("gap-x-4 w-64"):
        for key, pnl in sorted(by_key.items()):
            ui.label(key)
            ui.label(f"{pnl:,.2f}").classes(
                "text-green-500" if pnl >= 0 else "text-red-500"
            )


async def _render_attribution(
    summary: dict[str, Any], rows: list[dict[str, Any]]
) -> None:
    with ui.row().classes("gap-4 flex-wrap"):
        with ui.card().classes("w-56"):
            ui.label("Total P&L").classes("text-sm text-gray-500")
            ui.label(f"{summary['total_pnl']:,.2f}").classes("text-xl font-bold")
        with ui.card().classes("w-56"):
            ui.label("Win Rate").classes("text-sm text-gray-500")
            ui.label(_pct(summary["win_rate"])).classes("text-xl font-bold")
        with ui.card().classes("w-56"):
            ui.label("Positions").classes("text-sm text-gray-500")
            ui.label(str(summary["position_count"])).classes("text-xl font-bold")

    with ui.row().classes("gap-8 flex-wrap"):
        _attribution_mini_table("P&L by Strategy", summary["by_strategy"])
        _attribution_mini_table("P&L by LLM Verdict", summary["by_verdict"])

    ui.label("Positions").classes("text-lg font-bold mt-4")
    if not rows:
        ui.label("No paper positions from picks yet.").classes("text-gray-500")
        return
    ui.table(columns=_ATTRIBUTION_COLUMNS, rows=rows, row_key="symbol").classes("w-full")


async def _render_body() -> None:
    settings = get_settings()

    with BarStore(settings.duckdb_path) as store:
        outcomes = await evaluate_run_history(
            None, store, limit_runs=_LIMIT_RUNS, settings=settings
        )
    outcomes_summary = summarize_outcomes(outcomes)

    attribution_rows = await paper_attribution(settings=settings)
    attribution_summary_stats = attribution_summary(attribution_rows)

    ui.label("Pick Outcomes").classes("text-2xl font-bold")
    await _render_outcomes(outcomes_summary, outcomes)

    ui.label("Paper Attribution").classes("text-2xl font-bold mt-6")
    await _render_attribution(attribution_summary_stats, attribution_rows)


@ui.page("/history")
async def history_page() -> None:
    with page_frame("/history"):
        await _render_body()
