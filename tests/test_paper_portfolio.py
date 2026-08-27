"""Cash init/adjust, equity computation, and equity-snapshot upsert idempotency."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from sqlalchemy import select

from argus.config import AppSettings
from argus.data.prices.static_provider import synthetic_bars
from argus.data.store.duckdb_ohlcv import BarStore
from argus.db import async_session, init_db
from argus.db.models import PaperCash, PaperEquityPoint, PaperPosition
from argus.paper.portfolio import (
    IN_DOMAIN,
    US_DOMAIN,
    adjust_cash,
    cash_domain,
    ensure_all_cash_initialized,
    ensure_cash_initialized,
    get_cash,
    open_positions,
    portfolio_equity,
    reset_paper_account,
    snapshot_equity,
)


def _settings(tmp_path: Path) -> AppSettings:
    return AppSettings(data_dir=tmp_path, _env_file=None)  # type: ignore[call-arg]


def test_cash_domain_maps_markets_correctly() -> None:
    assert cash_domain("US_NASDAQ") == US_DOMAIN
    assert cash_domain("US_NYSE") == US_DOMAIN
    assert cash_domain("IN_NSE") == IN_DOMAIN


async def test_ensure_cash_initialized_seeds_from_settings(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    await init_db(settings)

    cash = await ensure_cash_initialized(US_DOMAIN, settings)
    assert cash == settings.paper.starting_cash_us

    cash_in = await ensure_cash_initialized(IN_DOMAIN, settings)
    assert cash_in == settings.paper.starting_cash_india


async def test_ensure_cash_initialized_is_idempotent(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    await init_db(settings)

    first = await ensure_cash_initialized(US_DOMAIN, settings)
    await adjust_cash(US_DOMAIN, -1_000.0, settings)
    second = await ensure_cash_initialized(US_DOMAIN, settings)

    assert second == first - 1_000.0

    async with async_session(settings) as session:
        rows = (
            (await session.execute(select(PaperCash).where(PaperCash.domain == US_DOMAIN)))
            .scalars()
            .all()
        )
        assert len(rows) == 1


async def test_get_cash_and_adjust_cash_round_trip(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    await init_db(settings)

    starting = await get_cash(US_DOMAIN, settings)
    new_balance = await adjust_cash(US_DOMAIN, -250.5, settings)
    assert new_balance == round(starting - 250.5, 2)

    reread = await get_cash(US_DOMAIN, settings)
    assert reread == new_balance


async def test_adjust_cash_rounds_to_two_decimals(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    await init_db(settings)
    await ensure_cash_initialized(US_DOMAIN, settings)

    new_balance = await adjust_cash(US_DOMAIN, -0.001, settings)
    assert new_balance == round(new_balance, 2)


async def test_ensure_all_cash_initialized_seeds_both_domains(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    await init_db(settings)

    await ensure_all_cash_initialized(settings)

    async with async_session(settings) as session:
        rows = (await session.execute(select(PaperCash))).scalars().all()
        domains = {r.domain for r in rows}
        assert domains == {US_DOMAIN, IN_DOMAIN}


async def test_open_positions_filters_by_market_and_closed_state(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    await init_db(settings)
    now = datetime.now(UTC)

    async with async_session(settings) as session:
        session.add(
            PaperPosition(
                symbol="AAPL", market="US_NASDAQ", qty=10, avg_price=100.0, opened_at=now
            )
        )
        session.add(
            PaperPosition(
                symbol="MSFT",
                market="US_NASDAQ",
                qty=5,
                avg_price=200.0,
                opened_at=now,
                closed_at=now,
            )
        )
        session.add(
            PaperPosition(symbol="RELI", market="IN_NSE", qty=20, avg_price=50.0, opened_at=now)
        )
        await session.commit()

    us_open = await open_positions("US_NASDAQ", settings)
    assert {p.symbol for p in us_open} == {"AAPL"}

    all_open = await open_positions(None, settings)
    assert {p.symbol for p in all_open} == {"AAPL", "RELI"}


def _store_with_bar(tmp_path: Path, market: str, symbol: str, close: float) -> BarStore:
    store = BarStore(tmp_path / "market_data.duckdb")
    today = date.today()  # noqa: DTZ011 -- matches the rest of the codebase's daily-cache boundary
    bars = synthetic_bars(
        n=5, start_price=close, seed=1, start=today - timedelta(days=4), trend=0.0
    )
    bars["close"][-1] = close
    bars["open"][-1] = close
    store.upsert_bars(market, symbol, bars)
    return store


async def test_portfolio_equity_is_cash_plus_marked_positions(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    await init_db(settings)
    await ensure_cash_initialized(US_DOMAIN, settings)
    await adjust_cash(US_DOMAIN, -1_000.0, settings)  # e.g. spent buying AAPL

    store = _store_with_bar(tmp_path, "US_NASDAQ", "AAPL", close=110.0)
    try:
        now = datetime.now(UTC)
        async with async_session(settings) as session:
            session.add(
                PaperPosition(
                    symbol="AAPL", market="US_NASDAQ", qty=10, avg_price=100.0, opened_at=now
                )
            )
            await session.commit()

        cash = await get_cash(US_DOMAIN, settings)
        equity = await portfolio_equity(US_DOMAIN, store, settings)
        assert equity == round(cash + 10 * 110.0, 2)
    finally:
        store.close()


async def test_snapshot_equity_writes_a_row(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    await init_db(settings)
    await ensure_cash_initialized(US_DOMAIN, settings)

    store = _store_with_bar(tmp_path, "US_NASDAQ", "AAPL", close=100.0)
    try:
        await snapshot_equity("US_NASDAQ", store, settings)

        async with async_session(settings) as session:
            rows = (
                await session.execute(
                    select(PaperEquityPoint).where(PaperEquityPoint.market == "US_NASDAQ")
                )
            ).scalars().all()
            assert len(rows) == 1
            assert rows[0].cash == settings.paper.starting_cash_us
            assert rows[0].positions_value == 0.0
    finally:
        store.close()


async def test_snapshot_equity_upsert_is_idempotent_on_date_and_market(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    await init_db(settings)
    await ensure_cash_initialized(US_DOMAIN, settings)

    store = _store_with_bar(tmp_path, "US_NASDAQ", "AAPL", close=100.0)
    try:
        await snapshot_equity("US_NASDAQ", store, settings)
        await adjust_cash(US_DOMAIN, -500.0, settings)
        await snapshot_equity("US_NASDAQ", store, settings)

        async with async_session(settings) as session:
            rows = (
                await session.execute(
                    select(PaperEquityPoint).where(PaperEquityPoint.market == "US_NASDAQ")
                )
            ).scalars().all()
            assert len(rows) == 1
            assert rows[0].cash == round(settings.paper.starting_cash_us - 500.0, 2)
    finally:
        store.close()


async def test_reset_paper_account_wipes_everything_and_reseeds_cash(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    await init_db(settings)
    now = datetime.now(UTC)

    await ensure_cash_initialized(US_DOMAIN, settings)
    await adjust_cash(US_DOMAIN, -1_000.0, settings)

    async with async_session(settings) as session:
        session.add(
            PaperPosition(symbol="AAPL", market="US_NASDAQ", qty=10, avg_price=100.0, opened_at=now)
        )
        session.add(
            PaperEquityPoint(
                date=now, market="US_NASDAQ", cash=1.0, positions_value=1.0, total_pnl=0.0
            )
        )
        await session.commit()

    await reset_paper_account(settings)

    async with async_session(settings) as session:
        assert (await session.execute(select(PaperPosition))).scalars().all() == []
        assert (await session.execute(select(PaperEquityPoint))).scalars().all() == []

    balance = await get_cash(US_DOMAIN, settings)
    assert balance == settings.paper.starting_cash_us
