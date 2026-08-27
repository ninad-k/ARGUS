"""refresh_bars populates the store from a provider and is idempotent on rerun."""

from datetime import date, timedelta
from pathlib import Path

from argus.data.prices.static_provider import StaticPriceProvider, synthetic_bars
from argus.data.store.duckdb_ohlcv import BarStore, refresh_bars
from argus.markets import Instrument


async def test_refresh_bars_populates_empty_store(tmp_path: Path) -> None:
    lookback_days = 10
    today = date.today()  # noqa: DTZ011
    start = today - timedelta(days=lookback_days)
    bars = synthetic_bars(n=lookback_days + 1, start_price=100.0, seed=11, start=start)

    provider = StaticPriceProvider({"AAPL": bars})
    inst = Instrument(symbol="AAPL", market_code="US_NASDAQ")

    with BarStore(tmp_path / "bars.duckdb") as store:
        added = await refresh_bars(store, provider, inst, lookback_days)
        assert added > 0

        stored = store.get_bars("US_NASDAQ", "AAPL")
        assert len(stored) == added


async def test_refresh_bars_second_run_adds_nothing(tmp_path: Path) -> None:
    lookback_days = 10
    today = date.today()  # noqa: DTZ011
    start = today - timedelta(days=lookback_days)
    bars = synthetic_bars(n=lookback_days + 1, start_price=100.0, seed=12, start=start)

    provider = StaticPriceProvider({"AAPL": bars})
    inst = Instrument(symbol="AAPL", market_code="US_NASDAQ")

    with BarStore(tmp_path / "bars.duckdb") as store:
        first = await refresh_bars(store, provider, inst, lookback_days)
        assert first > 0

        second = await refresh_bars(store, provider, inst, lookback_days)
        assert second == 0


async def test_refresh_bars_empty_provider_result_is_noop(tmp_path: Path) -> None:
    provider = StaticPriceProvider()  # no bars registered anywhere
    inst = Instrument(symbol="NOPE", market_code="US_NASDAQ")

    with BarStore(tmp_path / "bars.duckdb") as store:
        added = await refresh_bars(store, provider, inst, lookback_days=5)
        assert added == 0
        assert store.last_ts("US_NASDAQ", "NOPE") is None
