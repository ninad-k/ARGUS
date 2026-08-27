"""Full ``run_paper_cycle`` integration: day-1 pipeline picks get queued,
day-2 bars land in the store, running the cycle again fills the order and
writes equity snapshots for both days.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import select

from argus.config import AppSettings
from argus.data.prices.base import bars_from_columns
from argus.data.store.duckdb_ohlcv import BarStore
from argus.db import async_session, init_db
from argus.db.models import PaperEquityPoint, PaperOrder, PaperPosition
from argus.markets import Instrument
from argus.paper.engine import run_paper_cycle
from argus.paper.portfolio import US_DOMAIN, ensure_cash_initialized, get_cash
from argus.pipeline import ScreenReport
from argus.screener.base import Candidate
from argus.screener.runner import ScreenResult, persist_screen_result

_MARKET = "US_NASDAQ"


def _settings(tmp_path: Path) -> AppSettings:
    return AppSettings(data_dir=tmp_path, _env_file=None)  # type: ignore[call-arg]


def _put_bar(store: BarStore, symbol: str, day: date, open_: float, close: float) -> None:
    ts = np.array([np.datetime64(day.isoformat(), "s")], dtype="datetime64[s]")
    bars = bars_from_columns(
        ts,
        np.array([open_]),
        np.array([max(open_, close)]),
        np.array([min(open_, close)]),
        np.array([close]),
        np.array([1_000_000.0]),
    )
    store.upsert_bars(_MARKET, symbol, bars)


async def _report_for(settings: AppSettings, entry: float) -> ScreenReport:
    candidate = Candidate(
        instrument=Instrument(symbol="AAPL", market_code=_MARKET),
        strategy="momentum",
        score=90.0,
        direction="long",
        stage="breakout",
        reason="strong trend",
        entry=entry,
        stop=entry * 0.9,
        target=entry * 1.2,
    )
    result = ScreenResult(
        market_code=_MARKET,
        run_ts=datetime.now(UTC),
        universe_size=1,
        filtered_size=1,
        candidates=[candidate],
        top=[candidate],
    )
    run_id = await persist_screen_result(result, settings)
    return ScreenReport(
        result=result, run_id=run_id, bars_refreshed=0, symbols_failed=[], llm_used=False
    )


async def test_full_two_day_cycle_queues_then_fills_and_snapshots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    monkeypatch.setattr("argus.paper.engine.get_settings", lambda: settings)
    await init_db(settings)
    await ensure_cash_initialized(US_DOMAIN, settings)

    store = BarStore(tmp_path / "market_data.duckdb")
    try:
        day1 = date.today()  # noqa: DTZ011 -- matches the rest of the codebase's daily-cache boundary
        day2 = day1 + timedelta(days=1)

        # Day 1: some history exists (so the pick has something to work
        # from), the pipeline "picked" AAPL at entry=100.
        _put_bar(store, "AAPL", day1 - timedelta(days=1), open_=98.0, close=99.0)
        _put_bar(store, "AAPL", day1, open_=99.0, close=100.0)
        report_day1 = await _report_for(settings, entry=100.0)

        await run_paper_cycle(_MARKET, report_day1, store)

        # Nothing to fill yet (order was just queued this run) -- but a buy
        # order should now be pending, and today's equity snapshot exists.
        async with async_session(settings) as session:
            orders = (await session.execute(select(PaperOrder))).scalars().all()
            assert len(orders) == 1
            assert orders[0].status == "pending"
            assert orders[0].side == "buy"
            queued_qty = orders[0].qty

            equity_points_day1 = (
                (await session.execute(select(PaperEquityPoint))).scalars().all()
            )
            assert len(equity_points_day1) == 1

        cash_after_day1 = await get_cash(US_DOMAIN, settings)
        # No fill happened yet -- cash is still the full starting balance.
        assert cash_after_day1 == settings.paper.starting_cash_us

        # Day 2: a new session bar appears -- the pending order should fill
        # against its open. Re-persist a "new" report (as if the pipeline
        # ran again) so the cycle's queue step has something to do too.
        _put_bar(store, "AAPL", day2, open_=101.0, close=102.0)
        report_day2 = await _report_for(settings, entry=102.0)

        await run_paper_cycle(_MARKET, report_day2, store)

        async with async_session(settings) as session:
            filled_orders = (
                (
                    await session.execute(
                        select(PaperOrder).where(PaperOrder.status == "filled")
                    )
                )
                .scalars()
                .all()
            )
            assert len(filled_orders) == 1
            expected_fill = round(101.0 * (1 + settings.paper.slippage_bps / 10_000), 2)
            assert filled_orders[0].fill_price == expected_fill

            positions = (await session.execute(select(PaperPosition))).scalars().all()
            assert len(positions) == 1
            assert positions[0].qty == queued_qty

            equity_points_day2 = (
                (await session.execute(select(PaperEquityPoint))).scalars().all()
            )
            # Same market+date upserts in place -- but the second cycle's
            # snapshot happens on the same calendar day as the first in a
            # fast test run, so this should still be exactly one row.
            assert len(equity_points_day2) == 1

        cash_after_day2 = await get_cash(US_DOMAIN, settings)
        assert cash_after_day2 < cash_after_day1  # cash was debited by the fill
    finally:
        store.close()


async def test_run_paper_cycle_contains_a_failing_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure in one cycle step (here, fill_pending_orders) must not stop
    the later steps (queueing today's picks) from running."""
    settings = _settings(tmp_path)
    monkeypatch.setattr("argus.paper.engine.get_settings", lambda: settings)
    await init_db(settings)
    await ensure_cash_initialized(US_DOMAIN, settings)

    async def _boom(*args: object, **kwargs: object) -> int:
        raise RuntimeError("simulated failure")

    monkeypatch.setattr("argus.paper.engine.fill_pending_orders", _boom)

    store = BarStore(tmp_path / "market_data.duckdb")
    try:
        day1 = date.today()  # noqa: DTZ011
        _put_bar(store, "AAPL", day1, open_=99.0, close=100.0)
        report = await _report_for(settings, entry=100.0)

        await run_paper_cycle(_MARKET, report, store)

        async with async_session(settings) as session:
            orders = (await session.execute(select(PaperOrder))).scalars().all()
            assert len(orders) == 1  # queue step still ran despite the fill failure
    finally:
        store.close()
