"""Paper-trading engine: simulated order queuing, filling, and exit rules.

Everything here operates purely against ``BarStore`` bars and the
control-plane DB -- there is no broker/execution integration, and there
never will be. A "next_open" order queued today fills against the *next*
session's opening bar once it appears in the store (see
``fill_pending_orders``).

``run_paper_cycle`` is the single entry point the scheduler/smoke script
call: fill yesterday's orders against today's bars, apply stop/target exit
rules, queue new orders from today's picks, then snapshot equity. Each step
is exception-contained so one failure never blocks the others.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Literal

import numpy as np
import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from argus.config import get_settings
from argus.config.paper import PaperSettings
from argus.data.store.duckdb_ohlcv import BarStore
from argus.db import async_session
from argus.db.models import DailyPick, PaperCash, PaperOrder, PaperPosition
from argus.paper.portfolio import (
    cash_domain,
    ensure_cash_initialized,
    get_cash,
    open_positions,
    portfolio_equity,
    snapshot_equity,
)
from argus.paper.risk import PaperRiskGate
from argus.pipeline import ScreenReport

logger = structlog.get_logger(__name__)

# Pending orders older than this (calendar days since creation) that still
# haven't found a new session bar to fill against are cancelled outright,
# rather than left pending forever (e.g. a delisted/halted symbol).
_STALE_ORDER_DAYS = 5


@dataclass(frozen=True)
class OrderIntent:
    """A proposed simulated order, prior to risk validation."""

    symbol: str
    market_code: str
    side: Literal["buy", "sell"]
    qty: float
    ref_price: float
    pick_id: int | None


def _bar_date(bar: np.void) -> date:
    """The calendar date of a ``BAR_DTYPE`` row's ``ts`` field."""
    ts: datetime = bar["ts"].astype("datetime64[s]").astype(datetime)
    return ts.date()


async def queue_orders_for_picks(
    report: ScreenReport, *, settings: PaperSettings, store: BarStore
) -> list[int]:
    """Queue simulated buy orders for today's ``report.result.top`` picks.

    v1 only trades ``direction == "long"`` picks -- short selling is not
    simulated (paper trading is analysis-only; there is no borrow/margin
    model here). A pick with an LLM verdict of "avoid" is skipped. Qty is
    ``floor(equity * position_size_pct / 100 / entry)``, where equity is the
    cash+mark-to-market value of the pick's currency domain (see
    ``argus.paper.portfolio.portfolio_equity``); a pick sizing to zero, or
    failing the risk gate, is skipped (and logged) rather than raising.

    Cash/position-slot state is updated in-memory (not committed) as orders
    are queued within this call, so multiple picks in one run correctly
    compete for the same cash pool and max-positions slots instead of each
    being validated against a stale snapshot. Returns the created order ids.
    """
    market_code = report.result.market_code
    domain = cash_domain(market_code)
    app_settings = get_settings()

    cash = await get_cash(domain, app_settings)
    equity = await portfolio_equity(domain, store, app_settings)
    existing_positions = await open_positions(None, app_settings)
    open_symbols = frozenset((p.symbol, p.market) for p in existing_positions)
    open_count = len(existing_positions)

    created: list[int] = []
    now = datetime.now(UTC)
    async with async_session(app_settings) as session:
        for candidate in report.result.top:
            symbol = candidate.instrument.symbol

            if candidate.direction != "long":
                logger.info("paper.engine.skip_short", symbol=symbol)
                continue
            if candidate.llm_verdict is not None and candidate.llm_verdict.verdict == "avoid":
                logger.info("paper.engine.skip_avoid_verdict", symbol=symbol)
                continue
            if candidate.entry is None or candidate.entry <= 0:
                logger.info("paper.engine.skip_no_entry", symbol=symbol)
                continue

            qty = math.floor((equity * settings.position_size_pct / 100.0) / candidate.entry)
            if qty <= 0:
                logger.info("paper.engine.skip_zero_qty", symbol=symbol, entry=candidate.entry)
                continue

            intent = OrderIntent(
                symbol=symbol,
                market_code=market_code,
                side="buy",
                qty=float(qty),
                ref_price=candidate.entry,
                pick_id=None,
            )
            gate = PaperRiskGate(
                settings=settings,
                cash=cash,
                equity=equity,
                open_positions=open_symbols,
                open_position_count=open_count,
            )
            result = gate.validate(intent)
            if not result.ok:
                logger.info(
                    "paper.engine.risk_rejected",
                    symbol=symbol,
                    check=result.failed_check,
                    detail=result.detail,
                )
                continue

            pick_row = await session.scalar(
                select(DailyPick).where(
                    DailyPick.run_id == report.run_id, DailyPick.symbol == symbol
                )
            )
            order = PaperOrder(
                pick_id=pick_row.id if pick_row is not None else None,
                symbol=symbol,
                market=market_code,
                side="buy",
                qty=float(qty),
                order_type="next_open",
                status="pending",
                fill_price=None,
                slippage_bps=float(settings.slippage_bps),
                created_at=now,
                filled_at=None,
            )
            session.add(order)
            await session.flush()
            created.append(order.id)

            # Reserve this order's cash/slot so the next candidate in this
            # same run sees an up-to-date (not stale) view.
            cash -= qty * candidate.entry * (1 + settings.slippage_bps / 10_000)
            open_symbols = open_symbols | {(symbol, market_code)}
            open_count += 1

        await session.commit()
    return created


async def _adjust_cash_in_session(session: AsyncSession, domain: str, delta: float) -> None:
    """Adjust ``domain``'s cash within ``session``'s own (uncommitted)
    transaction -- unlike ``argus.paper.portfolio.adjust_cash``, which opens
    and commits its own session. Fills must stay in one transaction with the
    order/position updates: SQLite only allows one writer transaction at a
    time, so a nested session trying to commit while this call's outer
    session still has uncommitted work deadlocks ("database is locked")."""
    row = await session.scalar(select(PaperCash).where(PaperCash.domain == domain))
    if row is None:  # pragma: no cover -- callers ensure_cash_initialized first
        row = PaperCash(domain=domain, cash=0.0)
        session.add(row)
    row.cash = round(row.cash + delta, 2)


async def _settle_fill(
    session: AsyncSession,
    domain: str,
    order: PaperOrder,
    fill_price: float,
    now: datetime,
) -> None:
    """Apply one fill to its position (create/average-in on buy, reduce/
    close + realize P&L on sell) and to the domain cash ledger."""
    pos = await session.scalar(
        select(PaperPosition).where(
            PaperPosition.symbol == order.symbol,
            PaperPosition.market == order.market,
            PaperPosition.closed_at.is_(None),
        )
    )

    applied_qty: float
    if order.side == "buy":
        applied_qty = order.qty
        if pos is None:
            session.add(
                PaperPosition(
                    symbol=order.symbol,
                    market=order.market,
                    qty=order.qty,
                    avg_price=fill_price,
                    opened_at=now,
                    closed_at=None,
                    realized_pnl=None,
                )
            )
        else:
            new_qty = pos.qty + order.qty
            pos.avg_price = round((pos.avg_price * pos.qty + fill_price * order.qty) / new_qty, 2)
            pos.qty = new_qty
    else:
        if pos is None:
            logger.warning(
                "paper.engine.sell_without_position", symbol=order.symbol, market=order.market
            )
            return
        applied_qty = min(order.qty, pos.qty)
        realized = round((fill_price - pos.avg_price) * applied_qty, 2)
        pos.qty = round(pos.qty - applied_qty, 6)
        pos.realized_pnl = round((pos.realized_pnl or 0.0) + realized, 2)
        if pos.qty <= 0:
            pos.qty = 0.0
            pos.closed_at = now

    notional = round(applied_qty * fill_price, 2)
    delta = -notional if order.side == "buy" else notional
    await _adjust_cash_in_session(session, domain, delta)


async def fill_pending_orders(market_code: str, *, store: BarStore) -> int:
    """Fill pending "next_open" orders for ``market_code`` against new session bars.

    An order fills once a bar exists whose date is strictly after the
    order's ``created_at`` date -- the *next* trading session's open, at
    that bar's open price adjusted by slippage (buy: up, sell: down). If no
    such bar has appeared yet, the order is left pending, unless it's older
    than ``_STALE_ORDER_DAYS`` calendar days, in which case it's cancelled.
    Returns the number of orders filled.
    """
    app_settings = get_settings()
    domain = cash_domain(market_code)
    now = datetime.now(UTC)
    filled = 0

    await ensure_cash_initialized(domain, app_settings)
    async with async_session(app_settings) as session:
        pending = (
            await session.execute(
                select(PaperOrder).where(
                    PaperOrder.market == market_code, PaperOrder.status == "pending"
                )
            )
        ).scalars().all()

        for order in pending:
            bars = await asyncio.to_thread(store.get_bars, market_code, order.symbol, 10)
            created_date = order.created_at.date()
            new_bar = next((b for b in bars if _bar_date(b) > created_date), None)

            if new_bar is None:
                if (now.date() - created_date).days > _STALE_ORDER_DAYS:
                    order.status = "cancelled"
                continue

            open_price = float(new_bar["open"])
            slip = (order.slippage_bps or 0.0) / 10_000
            fill_price = round(
                open_price * (1 + slip) if order.side == "buy" else open_price * (1 - slip), 2
            )

            order.status = "filled"
            order.fill_price = fill_price
            order.filled_at = now
            await _settle_fill(session, domain, order, fill_price, now)
            filled += 1

        await session.commit()
    return filled


async def apply_exit_rules(market_code: str, *, store: BarStore) -> int:
    """Queue a full-qty sell for every open position whose originating pick's
    stop/target has been breached by the latest close.

    A position without a filled buy order carrying a ``pick_id``, or whose
    pick has neither a stop nor a target, is left alone -- v1's exit rules
    only apply to screener-originated positions. Returns the number of sell
    orders queued (a position that already has a pending sell is skipped,
    so this is idempotent across repeated calls on the same day).
    """
    app_settings = get_settings()
    queued = 0
    now = datetime.now(UTC)

    async with async_session(app_settings) as session:
        positions = (
            await session.execute(
                select(PaperPosition).where(
                    PaperPosition.market == market_code, PaperPosition.closed_at.is_(None)
                )
            )
        ).scalars().all()

        for pos in positions:
            buy_order = await session.scalar(
                select(PaperOrder)
                .where(
                    PaperOrder.symbol == pos.symbol,
                    PaperOrder.market == pos.market,
                    PaperOrder.side == "buy",
                    PaperOrder.status == "filled",
                    PaperOrder.pick_id.is_not(None),
                )
                .order_by(PaperOrder.filled_at.desc())
                .limit(1)
            )
            if buy_order is None or buy_order.pick_id is None:
                continue

            pick = await session.get(DailyPick, buy_order.pick_id)
            if pick is None or (pick.stop is None and pick.target is None):
                continue

            bars = await asyncio.to_thread(store.get_bars, market_code, pos.symbol, 1)
            if len(bars) == 0:
                continue
            latest_close = float(bars[-1]["close"])

            breached = (pick.stop is not None and latest_close <= pick.stop) or (
                pick.target is not None and latest_close >= pick.target
            )
            if not breached:
                continue

            already_pending = await session.scalar(
                select(PaperOrder).where(
                    PaperOrder.symbol == pos.symbol,
                    PaperOrder.market == pos.market,
                    PaperOrder.side == "sell",
                    PaperOrder.status == "pending",
                )
            )
            if already_pending is not None:
                continue

            session.add(
                PaperOrder(
                    pick_id=pick.id,
                    symbol=pos.symbol,
                    market=pos.market,
                    side="sell",
                    qty=pos.qty,
                    order_type="next_open",
                    status="pending",
                    fill_price=None,
                    slippage_bps=float(app_settings.paper.slippage_bps),
                    created_at=now,
                    filled_at=None,
                )
            )
            queued += 1

        await session.commit()
    return queued


async def run_paper_cycle(market_code: str, report: ScreenReport, store: BarStore) -> None:
    """One market's post-pipeline paper-trading cycle.

    Order matters: fill yesterday's queued orders against today's bars
    *before* queuing new orders from today's picks (otherwise a pick queued
    moments ago would immediately look "fillable" against today's own
    close/open data it was derived from). Each step is exception-contained
    -- a failure in one must never block the others or propagate to the
    caller (the scheduler / smoke script).
    """
    settings = get_settings()

    try:
        filled = await fill_pending_orders(market_code, store=store)
        logger.info("paper.engine.cycle.filled", market=market_code, count=filled)
    except Exception as exc:  # a bad fill pass must not block the rest of the cycle
        logger.error("paper.engine.cycle.fill_failed", market=market_code, error=str(exc))

    try:
        queued_exits = await apply_exit_rules(market_code, store=store)
        logger.info("paper.engine.cycle.exits_queued", market=market_code, count=queued_exits)
    except Exception as exc:
        logger.error("paper.engine.cycle.exit_rules_failed", market=market_code, error=str(exc))

    try:
        created = await queue_orders_for_picks(report, settings=settings.paper, store=store)
        logger.info("paper.engine.cycle.orders_queued", market=market_code, count=len(created))
    except Exception as exc:
        logger.error("paper.engine.cycle.queue_failed", market=market_code, error=str(exc))

    try:
        await snapshot_equity(market_code, store, settings)
        logger.info("paper.engine.cycle.equity_snapshotted", market=market_code)
    except Exception as exc:
        logger.error("paper.engine.cycle.snapshot_failed", market=market_code, error=str(exc))
