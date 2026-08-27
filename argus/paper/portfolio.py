"""Paper-trading cash + equity bookkeeping. No broker/execution code exists
here -- ``PaperCash`` is a purely simulated ledger seeded from
``PaperSettings`` and adjusted only by simulated fills (see
``argus.paper.engine``).

Cash is tracked per *currency domain* rather than per market: US_NYSE and
US_NASDAQ share one USD pool ("US"), IN_NSE has its own INR pool ("IN") --
see ``cash_domain``. ``PaperEquityPoint`` rows are still written per market
(``snapshot_equity``); for the two US markets that means each snapshot row
carries the *whole* US domain's cash alongside that one market's own
position value, since the domain's cash genuinely isn't split between them.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from sqlalchemy import delete, select

from argus.config import AppSettings, get_settings
from argus.data.store.duckdb_ohlcv import BarStore
from argus.db import async_session
from argus.db.models import PaperCash, PaperEquityPoint, PaperOrder, PaperPosition

US_DOMAIN = "US"
IN_DOMAIN = "IN"

_DOMAIN_MARKETS: dict[str, tuple[str, ...]] = {
    US_DOMAIN: ("US_NYSE", "US_NASDAQ"),
    IN_DOMAIN: ("IN_NSE",),
}


def cash_domain(market_code: str) -> str:
    """Map a market code to its cash currency-domain group."""
    return IN_DOMAIN if market_code == "IN_NSE" else US_DOMAIN


def _starting_cash(domain: str, settings: AppSettings) -> float:
    if domain == IN_DOMAIN:
        return settings.paper.starting_cash_india
    return settings.paper.starting_cash_us


async def ensure_cash_initialized(domain: str, settings: AppSettings | None = None) -> float:
    """Return ``domain``'s cash balance, seeding it from settings on first use."""
    settings = settings or get_settings()
    async with async_session(settings) as session:
        row = await session.scalar(select(PaperCash).where(PaperCash.domain == domain))
        if row is not None:
            return row.cash
        starting = round(_starting_cash(domain, settings), 2)
        session.add(PaperCash(domain=domain, cash=starting))
        await session.commit()
        return starting


async def ensure_all_cash_initialized(settings: AppSettings | None = None) -> None:
    settings = settings or get_settings()
    for domain in _DOMAIN_MARKETS:
        await ensure_cash_initialized(domain, settings)


async def get_cash(domain: str, settings: AppSettings | None = None) -> float:
    return await ensure_cash_initialized(domain, settings)


async def adjust_cash(domain: str, delta: float, settings: AppSettings | None = None) -> float:
    """Add ``delta`` (may be negative) to ``domain``'s cash balance; returns the new balance."""
    settings = settings or get_settings()
    await ensure_cash_initialized(domain, settings)
    async with async_session(settings) as session:
        row = await session.scalar(select(PaperCash).where(PaperCash.domain == domain))
        if row is None:  # pragma: no cover -- defensive; ensure_cash_initialized just ran
            row = PaperCash(domain=domain, cash=0.0)
            session.add(row)
        row.cash = round(row.cash + delta, 2)
        await session.commit()
        return row.cash


async def open_positions(
    market_code: str | None = None, settings: AppSettings | None = None
) -> list[PaperPosition]:
    """Open (``closed_at is None``) positions, optionally restricted to one market."""
    settings = settings or get_settings()
    async with async_session(settings) as session:
        stmt = select(PaperPosition).where(PaperPosition.closed_at.is_(None))
        if market_code is not None:
            stmt = stmt.where(PaperPosition.market == market_code)
        result = await session.execute(stmt)
        return list(result.scalars().all())


async def _positions_value(
    market_codes: tuple[str, ...], store: BarStore, settings: AppSettings
) -> float:
    total = 0.0
    for market_code in market_codes:
        for pos in await open_positions(market_code, settings):
            bars = await asyncio.to_thread(store.get_bars, pos.market, pos.symbol, 1)
            if len(bars) == 0:
                continue
            total += pos.qty * float(bars[-1]["close"])
    return total


async def portfolio_equity(
    domain: str, store: BarStore, settings: AppSettings | None = None
) -> float:
    """Cash + sum(qty * latest close) over every open position in ``domain``'s markets."""
    settings = settings or get_settings()
    cash = await get_cash(domain, settings)
    positions_value = await _positions_value(_DOMAIN_MARKETS[domain], store, settings)
    return round(cash + positions_value, 2)


async def snapshot_equity(
    market_code: str, store: BarStore, settings: AppSettings | None = None
) -> None:
    """Write (upsert on date+market) today's ``PaperEquityPoint`` for ``market_code``."""
    settings = settings or get_settings()
    domain = cash_domain(market_code)
    cash = round(await get_cash(domain, settings), 2)
    positions_value = round(await _positions_value((market_code,), store, settings), 2)
    total_pnl = round(cash + positions_value - _starting_cash(domain, settings), 2)

    today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    async with async_session(settings) as session:
        existing = await session.scalar(
            select(PaperEquityPoint).where(
                PaperEquityPoint.market == market_code, PaperEquityPoint.date == today
            )
        )
        if existing is not None:
            existing.cash = cash
            existing.positions_value = positions_value
            existing.total_pnl = total_pnl
        else:
            session.add(
                PaperEquityPoint(
                    date=today,
                    market=market_code,
                    cash=cash,
                    positions_value=positions_value,
                    total_pnl=total_pnl,
                )
            )
        await session.commit()


async def reset_paper_account(settings: AppSettings | None = None) -> None:
    """Wipe every paper order/position/cash/equity row and re-initialize cash
    from settings. Irreversible -- callers (the API/UI) are responsible for
    any user-facing confirmation; this function itself asks for none."""
    settings = settings or get_settings()
    async with async_session(settings) as session:
        await session.execute(delete(PaperOrder))
        await session.execute(delete(PaperPosition))
        await session.execute(delete(PaperEquityPoint))
        await session.execute(delete(PaperCash))
        await session.commit()
    await ensure_all_cash_initialized(settings)
