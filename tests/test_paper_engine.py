"""Paper engine: order sizing/risk-skip in ``queue_orders_for_picks``, fills
in ``fill_pending_orders`` (exact price/slippage, staleness, position/P&L
bookkeeping), and ``apply_exit_rules`` stop/target breach detection.

``argus.paper.engine`` resolves the control-plane DB via its own
module-level ``get_settings()`` (matching ``argus.pipeline``/
``argus.jobs.scheduler``'s own convention) rather than taking an
``AppSettings`` parameter, so every test here monkeypatches
``argus.paper.engine.get_settings`` to point at its ``tmp_path`` settings.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import select

from argus.config import AppSettings
from argus.config.paper import PaperSettings
from argus.data.prices.base import bars_from_columns
from argus.data.store.duckdb_ohlcv import BarStore
from argus.db import async_session, init_db
from argus.db.models import DailyPick, PaperOrder, PaperPosition
from argus.markets import Instrument
from argus.paper.engine import apply_exit_rules, fill_pending_orders, queue_orders_for_picks
from argus.paper.portfolio import US_DOMAIN, adjust_cash, ensure_cash_initialized, get_cash
from argus.pipeline import ScreenReport
from argus.screener.base import Candidate
from argus.screener.runner import ScreenResult, persist_screen_result

_MARKET = "US_NASDAQ"


def _settings(tmp_path: Path) -> AppSettings:
    return AppSettings(data_dir=tmp_path, _env_file=None)  # type: ignore[call-arg]


def _paper_settings(**overrides: object) -> PaperSettings:
    return PaperSettings(_env_file=None, **overrides)  # type: ignore[call-arg,arg-type]


def _store(tmp_path: Path) -> BarStore:
    return BarStore(tmp_path / "market_data.duckdb")


def _put_bar(store: BarStore, symbol: str, day: datetime, open_: float, close: float) -> None:
    ts = np.array([np.datetime64(day.date().isoformat(), "s")], dtype="datetime64[s]")
    bars = bars_from_columns(
        ts,
        np.array([open_]),
        np.array([max(open_, close)]),
        np.array([min(open_, close)]),
        np.array([close]),
        np.array([1_000_000.0]),
    )
    store.upsert_bars(_MARKET, symbol, bars)


def _candidate(
    symbol: str, entry: float, stop: float | None = None, target: float | None = None
) -> Candidate:
    return Candidate(
        instrument=Instrument(symbol=symbol, market_code=_MARKET),
        strategy="momentum",
        score=90.0,
        direction="long",
        stage="breakout",
        reason="strong trend",
        entry=entry,
        stop=stop,
        target=target,
    )


async def _persist_report(
    settings: AppSettings, candidates: list[Candidate], *, top_n: int | None = None
) -> ScreenReport:
    result = ScreenResult(
        market_code=_MARKET,
        run_ts=datetime.now(UTC),
        universe_size=len(candidates),
        filtered_size=len(candidates),
        candidates=candidates,
        top=candidates[:top_n] if top_n is not None else candidates,
    )
    run_id = await persist_screen_result(result, settings)
    return ScreenReport(
        result=result, run_id=run_id, bars_refreshed=0, symbols_failed=[], llm_used=False
    )


def _patch_engine_settings(monkeypatch: pytest.MonkeyPatch, settings: AppSettings) -> None:
    monkeypatch.setattr("argus.paper.engine.get_settings", lambda: settings)


async def test_queue_orders_sizes_qty_from_equity_and_position_pct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    _patch_engine_settings(monkeypatch, settings)
    await init_db(settings)
    await ensure_cash_initialized(US_DOMAIN, settings)  # starting_cash_us = 100_000

    report = await _persist_report(settings, [_candidate("AAPL", entry=100.0)])
    paper_settings = _paper_settings(position_size_pct=5.0, max_positions=10)

    with _store(tmp_path) as store:
        created = await queue_orders_for_picks(report, settings=paper_settings, store=store)

    assert len(created) == 1
    async with async_session(settings) as session:
        order = (await session.execute(select(PaperOrder))).scalar_one()
        # equity == starting cash (no positions) == 100_000; 5% of that /
        # entry(100) = 5000/100 = 50.
        assert order.qty == 50.0
        assert order.side == "buy"
        assert order.status == "pending"
        assert order.order_type == "next_open"
        assert order.symbol == "AAPL"
        pick = await session.scalar(select(DailyPick).where(DailyPick.symbol == "AAPL"))
        assert pick is not None
        assert order.pick_id == pick.id


async def test_queue_orders_skips_avoid_verdict_picks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from argus.advisor.pick_reviewer import PickVerdict

    settings = _settings(tmp_path)
    _patch_engine_settings(monkeypatch, settings)
    await init_db(settings)
    await ensure_cash_initialized(US_DOMAIN, settings)

    candidate = _candidate("BADCO", entry=50.0)
    candidate.llm_verdict = PickVerdict(
        symbol="BADCO", verdict="avoid", confidence=90, thesis="t", risks="r"
    )
    report = await _persist_report(settings, [candidate])
    paper_settings = _paper_settings()

    with _store(tmp_path) as store:
        created = await queue_orders_for_picks(report, settings=paper_settings, store=store)

    assert created == []
    async with async_session(settings) as session:
        orders = (await session.execute(select(PaperOrder))).scalars().all()
        assert orders == []


async def test_queue_orders_respects_max_positions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    _patch_engine_settings(monkeypatch, settings)
    await init_db(settings)
    await ensure_cash_initialized(US_DOMAIN, settings)

    report = await _persist_report(
        settings, [_candidate("AAPL", entry=100.0), _candidate("MSFT", entry=100.0)]
    )
    paper_settings = _paper_settings(position_size_pct=5.0, max_positions=1)

    with _store(tmp_path) as store:
        created = await queue_orders_for_picks(report, settings=paper_settings, store=store)

    # Only the first candidate should be queued -- the second trips
    # max_positions against the in-run reservation from the first.
    assert len(created) == 1
    async with async_session(settings) as session:
        orders = (await session.execute(select(PaperOrder))).scalars().all()
        assert len(orders) == 1
        assert orders[0].symbol == "AAPL"


async def test_fill_pending_orders_fills_at_next_open_with_slippage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    _patch_engine_settings(monkeypatch, settings)
    await init_db(settings)
    await ensure_cash_initialized(US_DOMAIN, settings)

    created_at = datetime.now(UTC) - timedelta(days=1)
    async with async_session(settings) as session:
        session.add(
            PaperOrder(
                symbol="AAPL",
                market=_MARKET,
                side="buy",
                qty=10.0,
                order_type="next_open",
                status="pending",
                slippage_bps=5.0,
                created_at=created_at,
            )
        )
        await session.commit()

    with _store(tmp_path) as store:
        _put_bar(store, "AAPL", created_at + timedelta(days=1), open_=100.0, close=105.0)
        filled = await fill_pending_orders(_MARKET, store=store)

    assert filled == 1
    async with async_session(settings) as session:
        order = (await session.execute(select(PaperOrder))).scalar_one()
        assert order.status == "filled"
        assert order.fill_price == round(100.0 * (1 + 5 / 10_000), 2)
        assert order.filled_at is not None


async def test_fill_pending_orders_leaves_pending_when_no_new_bar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    _patch_engine_settings(monkeypatch, settings)
    await init_db(settings)
    await ensure_cash_initialized(US_DOMAIN, settings)

    created_at = datetime.now(UTC)
    async with async_session(settings) as session:
        session.add(
            PaperOrder(
                symbol="AAPL",
                market=_MARKET,
                side="buy",
                qty=10.0,
                order_type="next_open",
                status="pending",
                slippage_bps=5.0,
                created_at=created_at,
            )
        )
        await session.commit()

    with _store(tmp_path) as store:
        # only a bar on/before the order's own creation date -- no new session yet
        _put_bar(store, "AAPL", created_at, open_=100.0, close=101.0)
        filled = await fill_pending_orders(_MARKET, store=store)

    assert filled == 0
    async with async_session(settings) as session:
        order = (await session.execute(select(PaperOrder))).scalar_one()
        assert order.status == "pending"


async def test_fill_pending_orders_cancels_stale_orders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    _patch_engine_settings(monkeypatch, settings)
    await init_db(settings)
    await ensure_cash_initialized(US_DOMAIN, settings)

    created_at = datetime.now(UTC) - timedelta(days=6)
    async with async_session(settings) as session:
        session.add(
            PaperOrder(
                symbol="AAPL",
                market=_MARKET,
                side="buy",
                qty=10.0,
                order_type="next_open",
                status="pending",
                slippage_bps=5.0,
                created_at=created_at,
            )
        )
        await session.commit()

    with _store(tmp_path) as store:  # no bars at all -> order never finds a new session
        filled = await fill_pending_orders(_MARKET, store=store)

    assert filled == 0
    async with async_session(settings) as session:
        order = (await session.execute(select(PaperOrder))).scalar_one()
        assert order.status == "cancelled"


async def test_fill_buy_creates_position_and_debits_cash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    _patch_engine_settings(monkeypatch, settings)
    await init_db(settings)
    starting_cash = await ensure_cash_initialized(US_DOMAIN, settings)

    created_at = datetime.now(UTC) - timedelta(days=1)
    async with async_session(settings) as session:
        session.add(
            PaperOrder(
                symbol="AAPL",
                market=_MARKET,
                side="buy",
                qty=10.0,
                order_type="next_open",
                status="pending",
                slippage_bps=0.0,
                created_at=created_at,
            )
        )
        await session.commit()

    with _store(tmp_path) as store:
        _put_bar(store, "AAPL", created_at + timedelta(days=1), open_=100.0, close=100.0)
        await fill_pending_orders(_MARKET, store=store)

    async with async_session(settings) as session:
        pos = (await session.execute(select(PaperPosition))).scalar_one()
        assert pos.qty == 10.0
        assert pos.avg_price == 100.0
        assert pos.closed_at is None

    new_cash = await get_cash(US_DOMAIN, settings)
    assert new_cash == round(starting_cash - 10.0 * 100.0, 2)


async def test_fill_sell_realizes_pnl_and_closes_position(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    _patch_engine_settings(monkeypatch, settings)
    await init_db(settings)
    starting_cash = await ensure_cash_initialized(US_DOMAIN, settings)

    now = datetime.now(UTC)
    async with async_session(settings) as session:
        session.add(
            PaperPosition(symbol="AAPL", market=_MARKET, qty=10.0, avg_price=100.0, opened_at=now)
        )
        await session.commit()

    created_at = now - timedelta(days=1)
    async with async_session(settings) as session:
        session.add(
            PaperOrder(
                symbol="AAPL",
                market=_MARKET,
                side="sell",
                qty=10.0,
                order_type="next_open",
                status="pending",
                slippage_bps=0.0,
                created_at=created_at,
            )
        )
        await session.commit()

    with _store(tmp_path) as store:
        _put_bar(store, "AAPL", created_at + timedelta(days=1), open_=110.0, close=110.0)
        await fill_pending_orders(_MARKET, store=store)

    async with async_session(settings) as session:
        pos = (await session.execute(select(PaperPosition))).scalar_one()
        assert pos.qty == 0.0
        assert pos.closed_at is not None
        assert pos.realized_pnl == round((110.0 - 100.0) * 10.0, 2)

    new_cash = await get_cash(US_DOMAIN, settings)
    assert new_cash == round(starting_cash + 10.0 * 110.0, 2)


async def _seed_position_with_pick(
    settings: AppSettings, *, stop: float | None, target: float | None
) -> None:
    now = datetime.now(UTC)
    async with async_session(settings) as session:
        pick = DailyPick(
            run_id=1,
            symbol="AAPL",
            market=_MARKET,
            strategy="momentum",
            score=90.0,
            stage="breakout",
            entry=100.0,
            stop=stop,
            target=target,
            features_json={},
            created_at=now,
        )
        session.add(pick)
        await session.flush()

        session.add(
            PaperOrder(
                pick_id=pick.id,
                symbol="AAPL",
                market=_MARKET,
                side="buy",
                qty=10.0,
                order_type="next_open",
                status="filled",
                fill_price=100.0,
                slippage_bps=0.0,
                created_at=now - timedelta(days=2),
                filled_at=now - timedelta(days=1),
            )
        )
        session.add(
            PaperPosition(symbol="AAPL", market=_MARKET, qty=10.0, avg_price=100.0, opened_at=now)
        )
        await session.commit()


async def test_apply_exit_rules_queues_sell_when_stop_breached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    _patch_engine_settings(monkeypatch, settings)
    await init_db(settings)
    await _seed_position_with_pick(settings, stop=90.0, target=120.0)

    with _store(tmp_path) as store:
        _put_bar(store, "AAPL", datetime.now(UTC), open_=85.0, close=85.0)  # below stop
        queued = await apply_exit_rules(_MARKET, store=store)

    assert queued == 1
    async with async_session(settings) as session:
        sell = await session.scalar(select(PaperOrder).where(PaperOrder.side == "sell"))
        assert sell is not None
        assert sell.qty == 10.0
        assert sell.status == "pending"


async def test_apply_exit_rules_queues_sell_when_target_breached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    _patch_engine_settings(monkeypatch, settings)
    await init_db(settings)
    await _seed_position_with_pick(settings, stop=90.0, target=120.0)

    with _store(tmp_path) as store:
        _put_bar(store, "AAPL", datetime.now(UTC), open_=125.0, close=125.0)  # above target
        queued = await apply_exit_rules(_MARKET, store=store)

    assert queued == 1
    async with async_session(settings) as session:
        sell = await session.scalar(select(PaperOrder).where(PaperOrder.side == "sell"))
        assert sell is not None


async def test_apply_exit_rules_skips_when_no_breach(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    _patch_engine_settings(monkeypatch, settings)
    await init_db(settings)
    await _seed_position_with_pick(settings, stop=90.0, target=120.0)

    with _store(tmp_path) as store:
        _put_bar(store, "AAPL", datetime.now(UTC), open_=105.0, close=105.0)  # within range
        queued = await apply_exit_rules(_MARKET, store=store)

    assert queued == 0


async def test_adjust_cash_available_for_manual_bookkeeping(tmp_path: Path) -> None:
    # Sanity check that adjust_cash (used internally by fills) round-trips.
    settings = _settings(tmp_path)
    await init_db(settings)
    starting = await ensure_cash_initialized(US_DOMAIN, settings)
    new_balance = await adjust_cash(US_DOMAIN, 100.0, settings)
    assert new_balance == round(starting + 100.0, 2)
