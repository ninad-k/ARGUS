"""Task 13 intraday-bar plumbing: ``BarStore`` intraday round-trip, every
non-yfinance provider's documented empty ``get_intraday_bars`` default, and
the pipeline persisting an "orderflow" key into ``DailyPick.features_json``
for a synthetic run."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest
from sqlalchemy import select

from argus.config import AppSettings
from argus.config.llm import LLMSettings
from argus.data.prices.composite import CompositePriceProvider
from argus.data.prices.nse_provider import NSEProvider
from argus.data.prices.static_provider import StaticPriceProvider, synthetic_bars
from argus.data.prices.tv_screener_provider import TVScreenerProvider
from argus.data.store.duckdb_ohlcv import BarStore
from argus.data.universe import StaticUniverseProvider
from argus.db import async_session
from argus.db.models import DailyPick
from argus.markets import IN_NSE, US_NASDAQ, Instrument
from argus.pipeline import run_daily_pipeline

_TODAY = date.today()  # noqa: DTZ011 -- matches refresh_bars' own daily-cache boundary


# ---------------------------------------------------------------------------
# BarStore intraday round-trip
# ---------------------------------------------------------------------------


def test_intraday_upsert_and_get_round_trip(tmp_path: Path) -> None:
    bars = synthetic_bars(n=10, start_price=100.0, seed=1, start=date(2026, 1, 2))
    with BarStore(tmp_path / "bars.duckdb") as store:
        added = store.upsert_intraday("US_NASDAQ", "AAPL", "15m", bars)
        assert added == 10

        result = store.get_intraday("US_NASDAQ", "AAPL", "15m")
        assert len(result) == 10
        assert list(result["ts"]) == list(bars["ts"])


def test_intraday_upsert_is_idempotent(tmp_path: Path) -> None:
    bars = synthetic_bars(n=5, start_price=50.0, seed=2, start=date(2026, 1, 2))
    with BarStore(tmp_path / "bars.duckdb") as store:
        store.upsert_intraday("US_NYSE", "JPM", "15m", bars)
        added_second = store.upsert_intraday("US_NYSE", "JPM", "15m", bars)
        assert added_second == 5
        assert len(store.get_intraday("US_NYSE", "JPM", "15m")) == 5


def test_intraday_last_n_returns_most_recent_ascending(tmp_path: Path) -> None:
    bars = synthetic_bars(n=20, start_price=50.0, seed=2, start=date(2026, 1, 2))
    with BarStore(tmp_path / "bars.duckdb") as store:
        store.upsert_intraday("US_NYSE", "JPM", "15m", bars)
        last_5 = store.get_intraday("US_NYSE", "JPM", "15m", last_n=5)
        assert len(last_5) == 5
        assert all(last_5["ts"][i] < last_5["ts"][i + 1] for i in range(4))
        full = store.get_intraday("US_NYSE", "JPM", "15m")
        assert list(last_5["ts"]) == list(full["ts"][-5:])


def test_intraday_isolated_by_interval(tmp_path: Path) -> None:
    bars_15m = synthetic_bars(n=5, start_price=100.0, seed=3, start=date(2026, 1, 2))
    bars_1h = synthetic_bars(n=3, start_price=100.0, seed=4, start=date(2026, 1, 2))
    with BarStore(tmp_path / "bars.duckdb") as store:
        store.upsert_intraday("US_NASDAQ", "AAPL", "15m", bars_15m)
        store.upsert_intraday("US_NASDAQ", "AAPL", "1h", bars_1h)

        assert len(store.get_intraday("US_NASDAQ", "AAPL", "15m")) == 5
        assert len(store.get_intraday("US_NASDAQ", "AAPL", "1h")) == 3


def test_intraday_empty_for_unknown_symbol(tmp_path: Path) -> None:
    with BarStore(tmp_path / "bars.duckdb") as store:
        assert len(store.get_intraday("US_NASDAQ", "NOPE", "15m")) == 0


# ---------------------------------------------------------------------------
# Provider defaults -- every provider but YFinanceProvider returns empty
# ---------------------------------------------------------------------------


async def test_static_provider_intraday_defaults_to_empty() -> None:
    daily = synthetic_bars(n=5, start_price=100.0, seed=1, start=date(2026, 1, 2))
    provider = StaticPriceProvider({"AAPL": daily})
    inst = Instrument(symbol="AAPL", market_code=US_NASDAQ.code)
    bars = await provider.get_intraday_bars(inst)
    assert len(bars) == 0


async def test_static_provider_intraday_serves_configured_bars() -> None:
    intraday = synthetic_bars(n=5, start_price=100.0, seed=1, start=date.today())  # noqa: DTZ011
    provider = StaticPriceProvider(intraday_by_symbol={"AAPL": intraday})
    inst = Instrument(symbol="AAPL", market_code=US_NASDAQ.code)
    bars = await provider.get_intraday_bars(inst, lookback_days=30)
    assert len(bars) == 5


async def test_nse_provider_intraday_defaults_to_empty() -> None:
    provider = NSEProvider()
    inst = Instrument(symbol="RELIANCE", market_code=IN_NSE.code)
    bars = await provider.get_intraday_bars(inst)
    assert len(bars) == 0


async def test_tv_screener_provider_intraday_defaults_to_empty() -> None:
    provider = TVScreenerProvider()
    inst = Instrument(symbol="AAPL", market_code=US_NASDAQ.code)
    bars = await provider.get_intraday_bars(inst)
    assert len(bars) == 0


async def test_composite_provider_intraday_falls_through_to_first_non_empty() -> None:
    intraday = synthetic_bars(n=5, start_price=100.0, seed=1, start=date.today())  # noqa: DTZ011
    empty_provider = StaticPriceProvider()
    serving_provider = StaticPriceProvider(intraday_by_symbol={"AAPL": intraday})
    composite = CompositePriceProvider([empty_provider, serving_provider])
    inst = Instrument(symbol="AAPL", market_code=US_NASDAQ.code)

    bars = await composite.get_intraday_bars(inst)

    assert len(bars) == 5


# ---------------------------------------------------------------------------
# Pipeline: orderflow annotation persists into DailyPick.features_json
# ---------------------------------------------------------------------------


def _settings(tmp_path: Path) -> AppSettings:
    return AppSettings(
        data_dir=tmp_path,
        llm=LLMSettings(enabled=False, _env_file=None),  # type: ignore[call-arg]
        _env_file=None,  # type: ignore[call-arg]
    )


async def test_pipeline_persists_orderflow_feature_for_top_picks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    monkeypatch.setattr("argus.pipeline.get_settings", lambda: settings)

    lookback = 299
    start = _TODAY - timedelta(days=lookback)
    bars_by_symbol = {
        "MOMO": synthetic_bars(n=300, start_price=100.0, seed=1, start=start, trend=0.006),
        "FLAT": synthetic_bars(n=300, start_price=80.0, seed=2, start=start, trend=0.0),
    }
    provider = StaticPriceProvider(bars_by_symbol)
    instruments = [Instrument(symbol=s, market_code=US_NASDAQ.code) for s in bars_by_symbol]
    universe_provider = StaticUniverseProvider({US_NASDAQ.code: instruments})

    report = await run_daily_pipeline(
        US_NASDAQ.code, provider=provider, universe_provider=universe_provider
    )

    assert report.run_id > 0
    assert report.result.top  # something cleared the screen

    async with async_session(settings) as session:
        picks = (
            await session.execute(select(DailyPick).where(DailyPick.run_id == report.run_id))
        ).scalars().all()

    top_symbols = {c.instrument.symbol for c in report.result.top}
    picks_by_symbol = {p.symbol: p for p in picks}
    for symbol in top_symbols:
        pick = picks_by_symbol[symbol]
        assert "orderflow" in pick.features_json
        orderflow = pick.features_json["orderflow"]
        assert isinstance(orderflow, dict)
        assert "gap_kind" in orderflow
