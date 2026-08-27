"""Instrument universe providers.

``TVUniverseProvider`` (Phase 2) wraps a live ``UniverseSource`` -- the
TradingView screener's liquidity ranking, see
``argus.data.prices.tv_screener_provider`` -- falling back to a static/seed
universe on any failure or empty result, so a screener run never blocks on a
flaky/unofficial upstream.
"""

import csv
from pathlib import Path
from typing import Protocol

import structlog
from sqlalchemy import select

from argus.config import AppSettings
from argus.data.prices.tv_screener_provider import UniverseSource
from argus.db import async_session
from argus.db.models import InstrumentRow
from argus.markets import IN_NSE, US_NASDAQ, US_NYSE, Instrument, Market

logger = structlog.get_logger(__name__)

_SEEDS_DIR = Path(__file__).parent / "seeds"
_FNO_FILE = "IN_NSE_fno.txt"
_US_MARKET_CODES = frozenset({US_NYSE.code, US_NASDAQ.code})

# Approximate NSE F&O lot sizes, in shares -- NSE revises these quarterly
# (based on a minimum-contract-value rule), so treat these as illustrative
# defaults for a paper/analysis-only screener, not a live-trading source of
# truth. Symbols not listed here (but still in ``IN_NSE_fno.txt``) fall back
# to lot_size=1.
_FNO_LOT_SIZES: dict[str, int] = {
    "RELIANCE": 250,
    "TCS": 175,
    "HDFCBANK": 550,
    "ICICIBANK": 700,
    "INFY": 400,
    "HINDUNILVR": 300,
    "ITC": 1600,
    "SBIN": 1500,
    "BHARTIARTL": 475,
    "KOTAKBANK": 400,
    "LT": 300,
    "AXISBANK": 625,
    "BAJFINANCE": 125,
    "ASIANPAINT": 200,
    "MARUTI": 50,
    "HCLTECH": 350,
    "SUNPHARMA": 350,
    "TITAN": 175,
    "ULTRACEMCO": 50,
    "WIPRO": 3000,
    "NESTLEIND": 25,
    "ADANIENT": 300,
    "ADANIPORTS": 1250,
    "ONGC": 3850,
    "NTPC": 1500,
    "POWERGRID": 2700,
    "TATAMOTORS": 1425,
    "TATASTEEL": 5500,
    "JSWSTEEL": 675,
    "COALINDIA": 2100,
    "BAJAJFINSV": 500,
    "HDFCLIFE": 1100,
    "SBILIFE": 750,
    "INDUSINDBK": 900,
    "GRASIM": 250,
    "DRREDDY": 125,
    "CIPLA": 650,
    "DIVISLAB": 100,
    "EICHERMOT": 175,
    "BRITANNIA": 200,
    "HEROMOTOCO": 300,
    "BPCL": 1800,
    "TECHM": 600,
    "UPL": 1400,
}


def _load_fno_symbols(seeds_dir: Path) -> frozenset[str]:
    """Read ``IN_NSE_fno.txt`` -- one uppercase symbol per line, ``#`` comments
    and blank lines ignored. Missing file -> empty set (no F&O flags set)."""
    path = seeds_dir / _FNO_FILE
    if not path.exists():
        return frozenset()
    with path.open(encoding="utf-8") as fh:
        return frozenset(
            line.strip().upper()
            for line in fh
            if line.strip() and not line.strip().startswith("#")
        )


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

        # US large caps virtually all have listed options -- flag every US
        # seed instrument. IN_NSE is pickier: only the curated F&O list gets
        # has_options/has_futures (+ its approximate lot size); everything
        # else in the NSE seed (mid/small caps with no derivatives segment)
        # keeps the has_options=False/lot_size=1 defaults.
        is_us_market = market.code in _US_MARKET_CODES
        fno_symbols = (
            _load_fno_symbols(self._seeds_dir) if market.code == IN_NSE.code else frozenset()
        )

        instruments: list[Instrument] = []
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                symbol = (row.get("symbol") or "").strip()
                if not symbol:
                    continue
                name = (row.get("name") or "").strip() or None
                sector = (row.get("sector") or "").strip() or None

                has_options = False
                has_futures = False
                lot_size = 1
                if is_us_market:
                    has_options = True
                elif symbol.upper() in fno_symbols:
                    has_options = True
                    has_futures = True
                    lot_size = _FNO_LOT_SIZES.get(symbol.upper(), 1)

                instruments.append(
                    Instrument(
                        symbol=symbol,
                        market_code=market.code,
                        name=name,
                        sector=sector,
                        has_options=has_options,
                        has_futures=has_futures,
                        lot_size=lot_size,
                    )
                )
        return instruments


class TVUniverseProvider:
    """Live top-liquid universe via a ``UniverseSource`` (the TradingView
    screener), falling back to ``fallback`` (typically ``SeedUniverseProvider``)
    on any failure or an empty result.
    """

    name = "tvscreener"

    def __init__(
        self, universe_source: UniverseSource, fallback: UniverseProvider, size: int
    ) -> None:
        self._universe_source = universe_source
        self._fallback = fallback
        self._size = size

    async def universe(self, market: Market) -> list[Instrument]:
        try:
            instruments = await self._universe_source.top_liquid(market, self._size)
        except Exception as exc:  # noqa: BLE001 -- never let a screener hiccup break the run
            logger.warning(
                "universe.tv_universe_provider.top_liquid_failed",
                market=market.code,
                error=str(exc),
            )
            instruments = []

        if instruments:
            return instruments

        logger.info("universe.tv_universe_provider.falling_back_to_seed", market=market.code)
        return await self._fallback.universe(market)


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
