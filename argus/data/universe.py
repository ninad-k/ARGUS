"""Instrument universe providers.

The tradingview-screener-backed dynamic universe (scanning the live market
for liquid names) is Phase 2 — not built here. This task only wires up
static/seed-file universes, which is enough to get the rest of the pipeline
running end-to-end offline.
"""

import csv
from pathlib import Path
from typing import Protocol

import structlog
from sqlalchemy import select

from argus.config import AppSettings
from argus.db import async_session
from argus.db.models import InstrumentRow
from argus.markets import Instrument, Market

logger = structlog.get_logger(__name__)

_SEEDS_DIR = Path(__file__).parent / "seeds"


class UniverseProvider(Protocol):
    """Resolves the tradable instrument universe for a market."""

    async def universe(self, market: Market) -> list[Instrument]:
        """Return the instruments to scan for ``market``."""
        ...


class StaticUniverseProvider:
    """Serves a fixed, caller-supplied universe. Primarily for tests."""

    name = "static"

    def __init__(self, instruments_by_market: dict[str, list[Instrument]] | None = None) -> None:
        self._instruments: dict[str, list[Instrument]] = instruments_by_market or {}

    def add(self, market_code: str, instruments: list[Instrument]) -> None:
        self._instruments[market_code] = instruments

    async def universe(self, market: Market) -> list[Instrument]:
        return list(self._instruments.get(market.code, []))


class SeedUniverseProvider:
    """Reads a fixed universe from ``argus/data/seeds/{market_code}.csv``.

    Each CSV has columns ``symbol,name,sector`` and covers a curated set of
    liquid large/mid-cap names for that market.
    """

    name = "seed"

    def __init__(self, seeds_dir: Path | None = None) -> None:
        self._seeds_dir = seeds_dir or _SEEDS_DIR

    async def universe(self, market: Market) -> list[Instrument]:
        path = self._seeds_dir / f"{market.code}.csv"
        if not path.exists():
            logger.warning(
                "universe.seed_provider.missing_file", market=market.code, path=str(path)
            )
            return []

        instruments: list[Instrument] = []
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                symbol = (row.get("symbol") or "").strip()
                if not symbol:
                    continue
                name = (row.get("name") or "").strip() or None
                sector = (row.get("sector") or "").strip() or None
                instruments.append(
                    Instrument(symbol=symbol, market_code=market.code, name=name, sector=sector)
                )
        return instruments


async def sync_instruments_to_db(
    instruments: list[Instrument], settings: AppSettings | None = None
) -> int:
    """Upsert ``instruments`` into the ``instruments`` table, keyed on (symbol, market).

    Returns the number of instruments synced.
    """
    if not instruments:
        return 0

    async with async_session(settings) as session:
        for inst in instruments:
            existing = await session.execute(
                select(InstrumentRow).where(
                    InstrumentRow.symbol == inst.symbol, InstrumentRow.market == inst.market_code
                )
            )
            row = existing.scalar_one_or_none()
            if row is None:
                session.add(
                    InstrumentRow(
                        symbol=inst.symbol,
                        market=inst.market_code,
                        name=inst.name,
                        sector=inst.sector,
                        lot_size=inst.lot_size,
                        tick_size=inst.tick_size,
                        has_options=inst.has_options,
                        has_futures=inst.has_futures,
                    )
                )
            else:
                row.name = inst.name
                row.sector = inst.sector
                row.lot_size = inst.lot_size
                row.tick_size = inst.tick_size
                row.has_options = inst.has_options
                row.has_futures = inst.has_futures
        await session.commit()
    return len(instruments)
