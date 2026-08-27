"""Local DuckDB cache of daily OHLCV bars.

DuckDB is a synchronous, embedded engine — ``BarStore`` itself is sync.
Async callers (the scheduler, refresh jobs) should wrap calls in
``asyncio.to_thread``; ``refresh_bars`` below does this for you.
"""

import asyncio
import threading
from datetime import date, datetime, timedelta
from pathlib import Path
from types import TracebackType

import duckdb
import numpy as np
import pandas as pd
import structlog
from numpy.typing import NDArray

from argus.data.prices.base import BAR_DTYPE, PriceDataProvider, bars_from_columns
from argus.markets import Instrument

logger = structlog.get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS daily_bars (
    market VARCHAR NOT NULL,
    symbol VARCHAR NOT NULL,
    ts TIMESTAMP NOT NULL,
    open DOUBLE NOT NULL,
    high DOUBLE NOT NULL,
    low DOUBLE NOT NULL,
    close DOUBLE NOT NULL,
    volume DOUBLE NOT NULL,
    PRIMARY KEY (market, symbol, ts)
)
"""

_INTRADAY_SCHEMA = """
CREATE TABLE IF NOT EXISTS intraday_bars (
    market VARCHAR NOT NULL,
    symbol VARCHAR NOT NULL,
    interval VARCHAR NOT NULL,
    ts TIMESTAMP NOT NULL,
    open DOUBLE NOT NULL,
    high DOUBLE NOT NULL,
    low DOUBLE NOT NULL,
    close DOUBLE NOT NULL,
    volume DOUBLE NOT NULL,
    PRIMARY KEY (market, symbol, interval, ts)
)
"""


class BarStore:
    """DuckDB-backed cache of daily OHLCV bars, keyed by (market, symbol, ts).

    A single ``duckdb.DuckDBPyConnection`` is not safe for concurrent use
    from multiple threads. Callers are expected to reach every query through
    ``asyncio.to_thread`` (see ``refresh_bars`` below) so that, from the
    caller's point of view, this class behaves as if it were async -- but
    that also means two ``to_thread`` calls against the *same* store (e.g.
    concurrent ``refresh_bars`` calls under a bounded-concurrency pipeline)
    can land on the connection from two different OS threads at once.
    ``_lock`` serializes those, since it's a ``threading.Lock`` rather than
    an ``asyncio.Lock`` (which would only protect against other coroutines
    on the same event loop thread, not against this).
    """

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = duckdb.connect(str(self._db_path))
        self._conn.execute(_SCHEMA)
        self._conn.execute(_INTRADAY_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "BarStore":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def upsert_bars(self, market_code: str, symbol: str, bars: NDArray[np.void]) -> int:
        """Insert or replace ``bars`` for ``(market_code, symbol)``. Returns row count."""
        if len(bars) == 0:
            return 0

        frame = pd.DataFrame(
            {
                "market": market_code,
                "symbol": symbol,
                "ts": bars["ts"].astype("datetime64[s]"),
                "open": bars["open"],
                "high": bars["high"],
                "low": bars["low"],
                "close": bars["close"],
                "volume": bars["volume"],
            }
        )
        with self._lock:
            self._conn.register("_argus_upsert_frame", frame)
            try:
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO daily_bars
                        (market, symbol, ts, open, high, low, close, volume)
                    SELECT market, symbol, ts, open, high, low, close, volume
                    FROM _argus_upsert_frame
                    """
                )
            finally:
                self._conn.unregister("_argus_upsert_frame")
        return int(len(frame))

    def get_bars(
        self, market_code: str, symbol: str, last_n: int | None = None
    ) -> NDArray[np.void]:
        """Return bars ascending by ``ts``. With ``last_n``, returns the most
        recent ``last_n`` rows (still ascending)."""
        if last_n is not None:
            query = """
                SELECT ts, open, high, low, close, volume FROM (
                    SELECT ts, open, high, low, close, volume FROM daily_bars
                    WHERE market = ? AND symbol = ?
                    ORDER BY ts DESC
                    LIMIT ?
                ) sub
                ORDER BY ts ASC
            """
            params: list[object] = [market_code, symbol, last_n]
        else:
            query = """
                SELECT ts, open, high, low, close, volume FROM daily_bars
                WHERE market = ? AND symbol = ?
                ORDER BY ts ASC
            """
            params = [market_code, symbol]

        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        if not rows:
            return np.zeros(0, dtype=BAR_DTYPE)

        ts = np.array([r[0] for r in rows], dtype="datetime64[s]")
        return bars_from_columns(
            ts,
            np.array([r[1] for r in rows], dtype=np.float64),
            np.array([r[2] for r in rows], dtype=np.float64),
            np.array([r[3] for r in rows], dtype=np.float64),
            np.array([r[4] for r in rows], dtype=np.float64),
            np.array([r[5] for r in rows], dtype=np.float64),
        )

    def upsert_intraday(
        self, market_code: str, symbol: str, interval: str, bars: NDArray[np.void]
    ) -> int:
        """Insert or replace intraday ``bars`` for ``(market_code, symbol, interval)``.
        Returns row count. Mirrors ``upsert_bars`` -- see its docstring."""
        if len(bars) == 0:
            return 0

        frame = pd.DataFrame(
            {
                "market": market_code,
                "symbol": symbol,
                "interval": interval,
                "ts": bars["ts"].astype("datetime64[s]"),
                "open": bars["open"],
                "high": bars["high"],
                "low": bars["low"],
                "close": bars["close"],
                "volume": bars["volume"],
            }
        )
        with self._lock:
            self._conn.register("_argus_upsert_intraday_frame", frame)
            try:
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO intraday_bars
                        (market, symbol, interval, ts, open, high, low, close, volume)
                    SELECT market, symbol, interval, ts, open, high, low, close, volume
                    FROM _argus_upsert_intraday_frame
                    """
                )
            finally:
                self._conn.unregister("_argus_upsert_intraday_frame")
        return int(len(frame))

    def get_intraday(
        self, market_code: str, symbol: str, interval: str, last_n: int | None = None
    ) -> NDArray[np.void]:
        """Return intraday bars ascending by ``ts``. Mirrors ``get_bars`` --
        see its docstring."""
        if last_n is not None:
            query = """
                SELECT ts, open, high, low, close, volume FROM (
                    SELECT ts, open, high, low, close, volume FROM intraday_bars
                    WHERE market = ? AND symbol = ? AND interval = ?
                    ORDER BY ts DESC
                    LIMIT ?
                ) sub
                ORDER BY ts ASC
            """
            params: list[object] = [market_code, symbol, interval, last_n]
        else:
            query = """
                SELECT ts, open, high, low, close, volume FROM intraday_bars
                WHERE market = ? AND symbol = ? AND interval = ?
                ORDER BY ts ASC
            """
            params = [market_code, symbol, interval]

        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        if not rows:
            return np.zeros(0, dtype=BAR_DTYPE)

        ts = np.array([r[0] for r in rows], dtype="datetime64[s]")
        return bars_from_columns(
            ts,
            np.array([r[1] for r in rows], dtype=np.float64),
            np.array([r[2] for r in rows], dtype=np.float64),
            np.array([r[3] for r in rows], dtype=np.float64),
            np.array([r[4] for r in rows], dtype=np.float64),
            np.array([r[5] for r in rows], dtype=np.float64),
        )

    def last_ts(self, market_code: str, symbol: str) -> datetime | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT MAX(ts) FROM daily_bars WHERE market = ? AND symbol = ?",
                [market_code, symbol],
            ).fetchone()
        if row is None or row[0] is None:
            return None
        value = row[0]
        return value if isinstance(value, datetime) else datetime.fromisoformat(str(value))

    def symbols(self, market_code: str) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT symbol FROM daily_bars WHERE market = ? ORDER BY symbol",
                [market_code],
            ).fetchall()
        return [str(r[0]) for r in rows]


async def refresh_bars(
    store: BarStore,
    provider: PriceDataProvider,
    inst: Instrument,
    lookback_days: int,
) -> int:
    """Fetch new bars for ``inst`` from the provider and upsert into ``store``.

    Fetches from ``last_ts + 1 day`` (or ``today - lookback_days`` if the
    store has no history yet) through today. Returns the number of rows
    upserted (0 if already up to date).
    """
    today = date.today()  # noqa: DTZ011 — daily cache boundary, not a precise instant
    last = await asyncio.to_thread(store.last_ts, inst.market_code, inst.symbol)
    if last is not None:
        start = last.date() + timedelta(days=1)
    else:
        start = today - timedelta(days=lookback_days)
    if start > today:
        return 0

    bars = await provider.get_daily_bars(inst, start, today)
    if len(bars) == 0:
        return 0
    return await asyncio.to_thread(store.upsert_bars, inst.market_code, inst.symbol, bars)


async def refresh_intraday(
    store: BarStore,
    provider: PriceDataProvider,
    inst: Instrument,
    interval: str = "15m",
    lookback_days: int = 30,
) -> int:
    """Fetch fresh intraday bars for ``inst`` and upsert into ``store``.

    Unlike ``refresh_bars``, this always re-fetches the full ``lookback_days``
    window rather than incrementally continuing from the store's last known
    timestamp -- intraday bars are only ever pulled for a handful of
    post-screen top candidates (Task 13), not a whole universe, so the extra
    cost of a full re-fetch each run is acceptable and keeps this simple.
    ``upsert_intraday`` is an ``INSERT OR REPLACE``, so re-fetching the same
    window is idempotent. Returns the number of rows upserted (0 if the
    provider has nothing, e.g. every non-yfinance provider today).
    """
    bars = await provider.get_intraday_bars(inst, interval=interval, lookback_days=lookback_days)
    if len(bars) == 0:
        return 0
    return await asyncio.to_thread(
        store.upsert_intraday, inst.market_code, inst.symbol, interval, bars
    )
