"""Paper account page (`/paper`): simulated portfolio, orders, and equity curve.

No broker/execution code exists here -- everything is a read view (plus a
guarded reset) over the paper-trading tables written by
``argus.paper.engine``/``argus.paper.portfolio``.
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog
from nicegui import ui

from argus.config import get_settings
from argus.data.store.duckdb_ohlcv import BarStore
from argus.db.models import PaperEquityPoint, PaperOrder, PaperPosition
from argus.paper.portfolio import IN_DOMAIN, US_DOMAIN, get_cash, reset_paper_account
from argus.ui.layout import page_frame
from argus.ui.queries import (
    open_paper_positions,
    paper_equity_points,
    recent_paper_orders,
    total_realized_pnl,
)

logger = structlog.get_logger(__name__)

_NA = "-"
_ORDER_ROWS_LIMIT = 50

_ORDER_COLUMNS: list[dict[str, Any]] = [
    {"name": "created", "label": "Created", "field": "created", "sortable": True},
    {"name": "symbol", "label": "Symbol", "field": "symbol", "sortable": True},
    {"name": "market", "label": "Market", "field": "market"},
    {"name": "side", "label": "Side", "field": "side"},
    {"name": "qty", "label": "Qty", "field": "qty"},
    {"name": "status", "label": "Status", "field": "status"},
    {"name": "fill_price", "label": "Fill Price", "field": "fill_price"},
]

_POSITION_COLUMNS: list[dict[str, Any]] = [
    {"name": "symbol", "label": "Symbol", "field": "symbol", "sortable": True},
    {"name": "market", "label": "Market", "field": "market"},
    {"name": "qty", "label": "Qty", "field": "qty"},
    {"name": "avg_price", "label": "Avg Price", "field": "avg_price"},
    {"name": "last_close", "label": "Last Close", "field": "last_close"},
    {"name": "unrealized_pnl", "label": "Unrealized P&L", "field": "unrealized_pnl"},
]


async def _latest_close(store: BarStore, market: str, symbol: str) -> float | None:
    bars = await asyncio.to_thread(store.get_bars, market, symbol, 1)
    if len(bars) == 0:
        return None
    return float(bars[-1]["close"])


async def _position_rows(positions: list[PaperPosition]) -> list[dict[str, Any]]:
    settings = get_settings()
    rows: list[dict[str, Any]] = []
    with BarStore(settings.duckdb_path) as store:
        for p in positions:
            last_close = await _latest_close(store, p.market, p.symbol)
            unrealized = (
                round((last_close - p.avg_price) * p.qty, 2) if last_close is not None else None
            )
            rows.append(
                {
                    "symbol": p.symbol,
                    "market": p.market,
                    "qty": p.qty,
                    "avg_price": round(p.avg_price, 2),
                    "last_close": round(last_close, 2) if last_close is not None else _NA,
                    "unrealized_pnl": unrealized if unrealized is not None else _NA,
                }
            )
    return rows


def _order_rows(orders: list[PaperOrder]) -> list[dict[str, Any]]:
    return [
        {
            "created": f"{o.created_at:%Y-%m-%d %H:%M}",
            "symbol": o.symbol,
            "market": o.market,
            "side": o.side,
            "qty": o.qty,
            "status": o.status,
            "fill_price": round(o.fill_price, 2) if o.fill_price is not None else _NA,
        }
        for o in orders
    ]


def _equity_chart_option(points: list[PaperEquityPoint]) -> dict[str, Any]:
    by_market: dict[str, list[list[Any]]] = {}
    for pt in points:
        by_market.setdefault(pt.market, []).append(
            [pt.date.strftime("%Y-%m-%d"), round(pt.cash + pt.positions_value, 2)]
        )
    return {
        "tooltip": {"trigger": "axis"},
        "legend": {},
        "xAxis": {"type": "category"},
        "yAxis": {"type": "value"},
        "series": [
            {"name": market, "type": "line", "data": series} for market, series in by_market.items()
        ],
    }


async def _render_body() -> None:
    settings = get_settings()
    cash_us = await get_cash(US_DOMAIN, settings)
    cash_in = await get_cash(IN_DOMAIN, settings)
    positions = await open_paper_positions(settings)
    orders = await recent_paper_orders(_ORDER_ROWS_LIMIT, settings)
    equity_points = await paper_equity_points(settings)
    realized = await total_realized_pnl(settings)

    with ui.row().classes("gap-4 flex-wrap"):
        with ui.card().classes("w-56"):
            ui.label("Cash (US)").classes("text-sm text-gray-500")
            ui.label(f"${cash_us:,.2f}").classes("text-xl font-bold")
        with ui.card().classes("w-56"):
            ui.label("Cash (India)").classes("text-sm text-gray-500")
            ui.label(f"₹{cash_in:,.2f}").classes("text-xl font-bold")
        with ui.card().classes("w-56"):
            ui.label("Open Positions").classes("text-sm text-gray-500")
            ui.label(str(len(positions))).classes("text-xl font-bold")
        with ui.card().classes("w-56"):
            ui.label("Realized P&L").classes("text-sm text-gray-500")
            ui.label(f"{realized:,.2f}").classes("text-xl font-bold")

    ui.label("Open Positions").classes("text-xl font-bold mt-4")
    if positions:
        ui.table(
            columns=_POSITION_COLUMNS, rows=await _position_rows(positions), row_key="symbol"
        ).classes("w-full")
    else:
        ui.label("No open positions.").classes("text-gray-500")

    ui.label("Recent Orders").classes("text-xl font-bold mt-4")
    if orders:
        ui.table(
            columns=_ORDER_COLUMNS, rows=_order_rows(orders), row_key="created"
        ).classes("w-full")
    else:
        ui.label("No orders yet.").classes("text-gray-500")

    ui.label("Equity Curve").classes("text-xl font-bold mt-4")
    if equity_points:
        ui.echart(_equity_chart_option(equity_points)).classes("w-full h-80")
    else:
        ui.label("No equity history yet.").classes("text-gray-500")


async def _refresh(container: ui.column) -> None:
    container.clear()
    with container:
        await _render_body()


async def _confirm_reset(container: ui.column) -> None:
    with ui.dialog() as dialog, ui.card():
        ui.label(
            "Reset the entire paper account? This permanently deletes all "
            "orders, positions, cash, and equity history."
        ).classes("w-80")
        with ui.row():
            ui.button("Cancel", on_click=dialog.close)

            async def _confirm() -> None:
                dialog.close()
                try:
                    await reset_paper_account()
                    ui.notify("Paper account reset", type="positive")
                except Exception as exc:  # a UI action must never crash the page
                    logger.error("ui.paper.reset_failed", error=str(exc))
                    ui.notify(f"Reset failed: {exc}", type="negative")
                await _refresh(container)

            ui.button("Reset", on_click=_confirm, color="negative")
    dialog.open()


@ui.page("/paper")
async def paper_page() -> None:
    content: ui.column
    with page_frame("/paper"):
        with ui.row().classes("items-center justify-between w-full"):
            ui.label("Paper Trading").classes("text-2xl font-bold")
            ui.button(
                "Reset Account", on_click=lambda: _confirm_reset(content), color="negative"
            )
        content = ui.column().classes("w-full gap-2")
        await _refresh(content)
