"""Core market abstraction: trading calendar and instrument identity."""

from dataclasses import dataclass
from datetime import date, time, timedelta
from zoneinfo import ZoneInfo


@dataclass(frozen=True, slots=True)
class Market:
    """A tradable market/exchange with its own calendar and session times."""

    code: str
    name: str
    timezone: ZoneInfo
    currency: str
    open_time: time
    close_time: time
    post_close_run: time
    holidays: frozenset[date]

    def is_trading_day(self, d: date) -> bool:
        return d.weekday() < 5 and d not in self.holidays

    def last_trading_day(self, before: date) -> date:
        """Most recent trading day strictly before ``before``."""
        cursor = before - timedelta(days=1)
        while not self.is_trading_day(cursor):
            cursor -= timedelta(days=1)
        return cursor

    def next_trading_day(self, after: date) -> date:
        """Next trading day strictly after ``after``."""
        cursor = after + timedelta(days=1)
        while not self.is_trading_day(cursor):
            cursor += timedelta(days=1)
        return cursor


@dataclass(frozen=True, slots=True)
class Instrument:
    """A single tradable symbol within a market."""

    symbol: str
    market_code: str
    name: str | None = None
    sector: str | None = None
    lot_size: int = 1
    tick_size: float = 0.01
    has_options: bool = False
    has_futures: bool = False
