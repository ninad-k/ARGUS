"""Paper-vs-pick attribution: which strategies and LLM verdicts actually
made (simulated) money.

Joins filled buy ``PaperOrder`` rows back to the ``DailyPick`` that spawned
them (via ``PaperOrder.pick_id``) and to the resulting ``PaperPosition``, so
each row answers "this pick, from this strategy, with this LLM verdict,
turned into this much P&L." Purely a read -- nothing here mutates an order,
pick, or position.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any, Literal

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from argus.config import AppSettings, get_settings
from argus.data.store.duckdb_ohlcv import BarStore
from argus.db import async_session
from argus.db.models import DailyPick, PaperOrder, PaperPosition

logger = structlog.get_logger(__name__)


async def _position_for_fill(
    session: AsyncSession, order: PaperOrder
) -> PaperPosition | None:
    """The ``PaperPosition`` a given filled buy order's shares landed in --
    the one open (or later closed) across ``order.filled_at``."""
    stmt = (
        select(PaperPosition)
        .where(
            PaperPosition.symbol == order.symbol,
            PaperPosition.market == order.market,
            PaperPosition.opened_at <= order.filled_at,
        )
        .order_by(PaperPosition.opened_at.desc())
        .limit(1)
    )
    position: PaperPosition | None = await session.scalar(stmt)
    return position


async def _closing_sell_price(
    session: AsyncSession, position: PaperPosition
) -> float | None:
    """The fill price of the sell order that closed ``position``, if any."""
    if position.closed_at is None:
        return None
    stmt = (
        select(PaperOrder)
        .where(
            PaperOrder.symbol == position.symbol,
            PaperOrder.market == position.market,
            PaperOrder.side == "sell",
            PaperOrder.status == "filled",
            PaperOrder.filled_at >= position.opened_at,
            PaperOrder.filled_at <= position.closed_at,
        )
        .order_by(PaperOrder.filled_at.desc())
        .limit(1)
    )
    sell = await session.scalar(stmt)
    return sell.fill_price if sell is not None else None


async def paper_attribution(
    *, store: BarStore | None = None, settings: AppSettings | None = None
) -> list[dict[str, Any]]:
    """One row per filled buy order that came from a pick: entry, exit
    (realized from the closing sell, or mark-to-market from the latest
    close for a still-open position), P&L, and the pick's LLM verdict.

    Never raises -- a bad row is logged and skipped; a total failure (e.g.
    the DB or bar store being unavailable) returns an empty list rather
    than propagating, since this is analytics, not part of any trading
    path.
    """
    resolved_settings = settings or get_settings()
    owns_store = store is None
    resolved_store = store if store is not None else BarStore(resolved_settings.duckdb_path)

    rows: list[dict[str, Any]] = []
    try:
        async with async_session(resolved_settings) as session:
            buy_orders = (
                await session.execute(
                    select(PaperOrder).where(
                        PaperOrder.side == "buy",
                        PaperOrder.status == "filled",
                        PaperOrder.pick_id.is_not(None),
                    )
                )
            ).scalars().all()

            for order in buy_orders:
                try:
                    if order.fill_price is None or order.pick_id is None:
                        continue
                    pick = await session.get(DailyPick, order.pick_id)
                    if pick is None:
                        continue

                    position = await _position_for_fill(session, order)

                    exit_price: float | None = None
                    status: Literal["open", "closed"] = "open"
                    if position is not None:
                        exit_price = await _closing_sell_price(session, position)
                        if exit_price is not None:
                            status = "closed"

                    if exit_price is None:
                        bars = await asyncio.to_thread(
                            resolved_store.get_bars, order.market, order.symbol, 1
                        )
                        if len(bars) > 0:
                            exit_price = float(bars[-1]["close"])

                    pnl: float | None = None
                    pnl_pct: float | None = None
                    if exit_price is not None:
                        pnl = round((exit_price - order.fill_price) * order.qty, 2)
                        pnl_pct = round(
                            (exit_price - order.fill_price) / order.fill_price * 100, 2
                        )

                    llm_verdict = None
                    if pick.llm_verdict_json:
                        llm_verdict = pick.llm_verdict_json.get("verdict")

                    rows.append(
                        {
                            "symbol": order.symbol,
                            "market": order.market,
                            "strategy": pick.strategy,
                            "picked_at": pick.created_at.date().isoformat(),
                            "fill_price": order.fill_price,
                            "qty": order.qty,
                            "exit_price": exit_price,
                            "pnl": pnl,
                            "pnl_pct": pnl_pct,
                            "llm_verdict": llm_verdict,
                            "status": status,
                        }
                    )
                except Exception as exc:  # a bad row must never abort the batch
                    logger.warning(
                        "analysis.attribution.row_failed", order_id=order.id, error=str(exc)
                    )
    except Exception as exc:  # analytics must never raise into a caller
        logger.warning("analysis.attribution.failed", error=str(exc))
    finally:
        if owns_store:
            resolved_store.close()

    return rows


def attribution_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Total/realized-or-unrealized P&L, win rate, and P&L broken down by
    strategy and by LLM verdict -- answers "which strategies and verdicts
    actually make paper money."

    A fused strategy label (e.g. ``"momentum+breakout"``) is attributed to
    each component strategy, matching ``argus.analysis.outcomes.summarize_outcomes``.
    """
    priced = [r for r in rows if r["pnl"] is not None]
    total_pnl = round(sum(r["pnl"] for r in priced), 2) if priced else 0.0
    wins = [r for r in priced if r["pnl"] > 0]
    win_rate = round(len(wins) / len(priced), 4) if priced else 0.0

    by_strategy: dict[str, float] = defaultdict(float)
    for r in priced:
        for strat in str(r["strategy"]).split("+"):
            if strat:
                by_strategy[strat] += r["pnl"]

    by_verdict: dict[str, float] = defaultdict(float)
    for r in priced:
        verdict = r["llm_verdict"] or "none"
        by_verdict[verdict] += r["pnl"]

    return {
        "total_pnl": total_pnl,
        "win_rate": win_rate,
        "position_count": len(rows),
        "priced_count": len(priced),
        "by_strategy": {k: round(v, 2) for k, v in sorted(by_strategy.items())},
        "by_verdict": {k: round(v, 2) for k, v in sorted(by_verdict.items())},
    }
