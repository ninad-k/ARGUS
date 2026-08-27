"""Paper-trading REST endpoints -- read-only views over the simulated
portfolio, plus a dangerous full reset. No broker/execution code exists
here or anywhere else in ARGUS.
"""

from __future__ import annotations

from fastapi import APIRouter, Query
from sqlalchemy import select

from argus.api.schemas import (
    PaperEquityPointOut,
    PaperEquityResponse,
    PaperOrderOut,
    PaperOrdersResponse,
    PaperPositionOut,
    PaperPositionsResponse,
    PaperResetResponse,
)
from argus.config import get_settings
from argus.db import async_session
from argus.db.models import PaperEquityPoint, PaperOrder, PaperPosition
from argus.paper.portfolio import reset_paper_account

router = APIRouter(prefix="/api/v1/paper", tags=["paper"])


def _order_out(o: PaperOrder) -> PaperOrderOut:
    return PaperOrderOut(
        id=o.id,
        pick_id=o.pick_id,
        symbol=o.symbol,
        market=o.market,
        side=o.side,
        qty=o.qty,
        order_type=o.order_type,
        status=o.status,
        fill_price=o.fill_price,
        slippage_bps=o.slippage_bps,
        created_at=o.created_at,
        filled_at=o.filled_at,
    )


def _position_out(p: PaperPosition) -> PaperPositionOut:
    return PaperPositionOut(
        id=p.id,
        symbol=p.symbol,
        market=p.market,
        qty=p.qty,
        avg_price=p.avg_price,
        opened_at=p.opened_at,
        closed_at=p.closed_at,
        realized_pnl=p.realized_pnl,
    )


def _equity_out(e: PaperEquityPoint) -> PaperEquityPointOut:
    return PaperEquityPointOut(
        id=e.id,
        date=e.date,
        market=e.market,
        cash=e.cash,
        positions_value=e.positions_value,
        total_pnl=e.total_pnl,
    )


@router.get("/positions", response_model=PaperPositionsResponse)
async def get_positions(open_only: bool = Query(default=True)) -> PaperPositionsResponse:
    settings = get_settings()
    async with async_session(settings) as session:
        stmt = select(PaperPosition)
        if open_only:
            stmt = stmt.where(PaperPosition.closed_at.is_(None))
        result = await session.execute(stmt.order_by(PaperPosition.opened_at.desc()))
        positions = list(result.scalars().all())
    return PaperPositionsResponse(positions=[_position_out(p) for p in positions])


@router.get("/orders", response_model=PaperOrdersResponse)
async def get_orders(status: str | None = Query(default=None)) -> PaperOrdersResponse:
    settings = get_settings()
    async with async_session(settings) as session:
        stmt = select(PaperOrder)
        if status is not None:
            stmt = stmt.where(PaperOrder.status == status)
        result = await session.execute(stmt.order_by(PaperOrder.created_at.desc()))
        orders = list(result.scalars().all())
    return PaperOrdersResponse(orders=[_order_out(o) for o in orders])


@router.get("/equity", response_model=PaperEquityResponse)
async def get_equity(market: str | None = Query(default=None)) -> PaperEquityResponse:
    settings = get_settings()
    async with async_session(settings) as session:
        stmt = select(PaperEquityPoint)
        if market is not None:
            stmt = stmt.where(PaperEquityPoint.market == market)
        result = await session.execute(stmt.order_by(PaperEquityPoint.date.asc()))
        points = list(result.scalars().all())
    return PaperEquityResponse(points=[_equity_out(e) for e in points])


@router.post("/reset", response_model=PaperResetResponse)
async def post_reset() -> PaperResetResponse:
    """Wipe every paper order/position/cash/equity row and re-initialize cash
    from settings. DANGEROUS -- irreversible and confirm-free at this layer;
    the UI (``argus.ui.paper``) gates it behind a confirmation dialog."""
    settings = get_settings()
    await reset_paper_account(settings)
    return PaperResetResponse(ok=True, detail="paper account reset")
