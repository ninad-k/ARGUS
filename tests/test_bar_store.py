"""BarStore upsert idempotency, ordering, and lookup helpers against a tmp DuckDB."""

from datetime import date
from pathlib import Path

from argus.data.prices.static_provider import synthetic_bars
from argus.data.store.duckdb_ohlcv import BarStore


def test_upsert_and_reupsert_is_idempotent(tmp_path: Path) -> None:
    bars = synthetic_bars(n=10, start_price=100.0, seed=1, start=date(2026, 1, 2))
    with BarStore(tmp_path / "bars.duckdb") as store:
        added_first = store.upsert_bars("US_NASDAQ", "AAPL", bars)
        assert added_first == 10

        added_second = store.upsert_bars("US_NASDAQ", "AAPL", bars)
        assert added_second == 10  # INSERT OR REPLACE overwrites, doesn't duplicate

        result = store.get_bars("US_NASDAQ", "AAPL")
        assert len(result) == 10


def test_get_bars_last_n_returns_most_recent_ascending(tmp_path: Path) -> None:
    bars = synthetic_bars(n=20, start_price=50.0, seed=2, start=date(2026, 1, 2))
    with BarStore(tmp_path / "bars.duckdb") as store:
        store.upsert_bars("US_NYSE", "JPM", bars)

        last_5 = store.get_bars("US_NYSE", "JPM", last_n=5)
        assert len(last_5) == 5
        # Ascending order.
        assert all(last_5["ts"][i] < last_5["ts"][i + 1] for i in range(4))
        # These are the most recent 5 rows from the full set.
        full = store.get_bars("US_NYSE", "JPM")
        assert list(last_5["ts"]) == list(full["ts"][-5:])


def test_last_ts_returns_none_when_no_data(tmp_path: Path) -> None:
    with BarStore(tmp_path / "bars.duckdb") as store:
        assert store.last_ts("US_NYSE", "NOPE") is None


def test_last_ts_returns_max_timestamp(tmp_path: Path) -> None:
    bars = synthetic_bars(n=5, start_price=200.0, seed=3, start=date(2026, 1, 2))
    with BarStore(tmp_path / "bars.duckdb") as store:
        store.upsert_bars("IN_NSE", "RELIANCE", bars)
        last = store.last_ts("IN_NSE", "RELIANCE")
        assert last is not None
        assert last.date() == date(2026, 1, 6)


def test_symbols_lists_distinct_symbols_for_market(tmp_path: Path) -> None:
    bars = synthetic_bars(n=3, start_price=10.0, seed=4, start=date(2026, 1, 2))
    with BarStore(tmp_path / "bars.duckdb") as store:
        store.upsert_bars("US_NASDAQ", "AAPL", bars)
        store.upsert_bars("US_NASDAQ", "MSFT", bars)
        store.upsert_bars("US_NYSE", "JPM", bars)

        assert store.symbols("US_NASDAQ") == ["AAPL", "MSFT"]
        assert store.symbols("US_NYSE") == ["JPM"]


def test_get_bars_empty_for_unknown_symbol(tmp_path: Path) -> None:
    with BarStore(tmp_path / "bars.duckdb") as store:
        result = store.get_bars("US_NASDAQ", "NOPE")
        assert len(result) == 0
