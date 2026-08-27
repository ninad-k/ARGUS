"""Seed CSV parsing and DB round-trip for the instrument universe."""

from pathlib import Path

from sqlalchemy import select

from argus.config import AppSettings
from argus.data.universe import SeedUniverseProvider, StaticUniverseProvider, sync_instruments_to_db
from argus.db import async_session, init_db
from argus.db.models import InstrumentRow
from argus.markets import IN_NSE, US_NASDAQ, US_NYSE, Instrument


async def test_seed_universe_parses_for_all_markets() -> None:
    provider = SeedUniverseProvider()
    for market in (US_NASDAQ, US_NYSE, IN_NSE):
        instruments = await provider.universe(market)
        assert len(instruments) >= 40, f"{market.code} has only {len(instruments)} instruments"
        assert all(inst.market_code == market.code for inst in instruments)
        assert all(inst.symbol for inst in instruments)
        # No duplicate symbols within a market's seed file.
        symbols = [inst.symbol for inst in instruments]
        assert len(symbols) == len(set(symbols))


async def test_seed_universe_missing_file_returns_empty(tmp_path: Path) -> None:
    provider = SeedUniverseProvider(seeds_dir=tmp_path)
    result = await provider.universe(US_NASDAQ)
    assert result == []


async def test_static_universe_provider_returns_added_instruments() -> None:
    provider = StaticUniverseProvider()
    inst = Instrument(symbol="AAPL", market_code=US_NASDAQ.code, name="Apple Inc.")
    provider.add(US_NASDAQ.code, [inst])

    result = await provider.universe(US_NASDAQ)
    assert result == [inst]

    other = await provider.universe(US_NYSE)
    assert other == []


async def test_sync_instruments_to_db_round_trips(tmp_path: Path) -> None:
    settings = AppSettings(data_dir=tmp_path, _env_file=None)  # type: ignore[call-arg]
    await init_db(settings)

    instruments = [
        Instrument(
            symbol="AAPL", market_code=US_NASDAQ.code, name="Apple Inc.", sector="Technology"
        ),
        Instrument(
            symbol="JPM", market_code=US_NYSE.code, name="JPMorgan Chase & Co.", sector="Financials"
        ),
    ]
    synced = await sync_instruments_to_db(instruments, settings)
    assert synced == 2

    async with async_session(settings) as session:
        result = await session.execute(select(InstrumentRow).order_by(InstrumentRow.symbol))
        rows = result.scalars().all()

    assert [r.symbol for r in rows] == ["AAPL", "JPM"]
    assert rows[0].sector == "Technology"


async def test_sync_instruments_to_db_updates_existing_row(tmp_path: Path) -> None:
    settings = AppSettings(data_dir=tmp_path, _env_file=None)  # type: ignore[call-arg]
    await init_db(settings)

    inst = Instrument(
        symbol="AAPL", market_code=US_NASDAQ.code, name="Apple Inc.", sector="Technology"
    )
    await sync_instruments_to_db([inst], settings)

    updated = Instrument(
        symbol="AAPL", market_code=US_NASDAQ.code, name="Apple Inc.", sector="Consumer Tech"
    )
    await sync_instruments_to_db([updated], settings)

    async with async_session(settings) as session:
        result = await session.execute(select(InstrumentRow))
        rows = result.scalars().all()

    assert len(rows) == 1  # updated in place, not duplicated
    assert rows[0].sector == "Consumer Tech"
