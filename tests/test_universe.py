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


async def test_seed_universe_us_markets_all_flagged_has_options() -> None:
    provider = SeedUniverseProvider()
    for market in (US_NASDAQ, US_NYSE):
        instruments = await provider.universe(market)
        assert instruments
        assert all(inst.has_options for inst in instruments)
        assert all(not inst.has_futures for inst in instruments)
        assert all(inst.lot_size == 1 for inst in instruments)


async def test_seed_universe_in_nse_fno_symbols_flagged_with_lot_size() -> None:
    provider = SeedUniverseProvider()
    instruments = await provider.universe(IN_NSE)
    by_symbol = {inst.symbol: inst for inst in instruments}

    reliance = by_symbol["RELIANCE"]
    assert reliance.has_options is True
    assert reliance.has_futures is True
    assert reliance.lot_size == 250

    # At least one F&O name and one non-F&O name are present, so the split
    # actually exercises both branches (not every IN_NSE seed symbol is F&O).
    fno = [inst for inst in instruments if inst.has_options]
    non_fno = [inst for inst in instruments if not inst.has_options]
    assert len(fno) >= 30
    assert non_fno
    assert all(inst.has_futures is False and inst.lot_size == 1 for inst in non_fno)


async def test_seed_universe_in_nse_fno_symbol_without_lot_size_entry_defaults_to_one(
    tmp_path: Path,
) -> None:
    """A symbol present in the F&O list but absent from ``_FNO_LOT_SIZES``
    still gets has_options/has_futures -- just with the lot_size=1 default."""
    (tmp_path / "IN_NSE.csv").write_text(
        "symbol,name,sector\nZZZFNO,ZZZ Corp,Industrials\n", encoding="utf-8"
    )
    (tmp_path / "IN_NSE_fno.txt").write_text("ZZZFNO\n", encoding="utf-8")

    provider = SeedUniverseProvider(seeds_dir=tmp_path)
    instruments = await provider.universe(IN_NSE)

    assert len(instruments) == 1
    assert instruments[0].has_options is True
    assert instruments[0].has_futures is True
    assert instruments[0].lot_size == 1


async def test_seed_universe_in_nse_missing_fno_file_sets_no_flags(tmp_path: Path) -> None:
    (tmp_path / "IN_NSE.csv").write_text(
        "symbol,name,sector\nRELIANCE,Reliance Industries Ltd.,Energy\n", encoding="utf-8"
    )
    provider = SeedUniverseProvider(seeds_dir=tmp_path)
    instruments = await provider.universe(IN_NSE)

    assert len(instruments) == 1
    assert instruments[0].has_options is False
    assert instruments[0].has_futures is False
    assert instruments[0].lot_size == 1


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
