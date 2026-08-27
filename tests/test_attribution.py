"""``paper_attribution``'s row-building (closed and still-open positions) and
``attribution_summary``'s P&L math, by-strategy, and by-verdict breakdowns.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from argus.analysis.attribution import attribution_summary, paper_attribution
from argus.config import AppSettings
from argus.data.prices.base import bars_from_columns
from argus.data.store.duckdb_ohlcv import BarStore
from argus.db import async_session, init_db
from argus.db.models import DailyPick, PaperOrder, PaperPosition

_MARKET = "US_NASDAQ"


def _settings(tmp_path: Path) -> AppSettings:
    return AppSettings(data_dir=tmp_path, _env_file=None)  # type: ignore[call-arg]


def _store(tmp_path: Path) -> BarStore:
    return BarStore(tmp_path / "market_data.duckdb")


def _put_bar(store: BarStore, symbol: str, day: datetime, close: float) -> None:
    ts = np.array([np.datetime64(day.date().isoformat(), "s")], dtype="datetime64[s]")
    bars = bars_from_columns(
        ts,
        np.array([close]),
        np.array([close]),
        np.array([close]),
        np.array([close]),
        np.array([1_000_000.0]),
    )
    store.upsert_bars(_MARKET, symbol, bars)


async def _add_pick(
    settings: AppSettings,
    *,
    symbol: str,
    strategy: str,
    verdict: str | None = None,
) -> int:
    async with async_session(settings) as session:
        pick = DailyPick(
            run_id=1,
            symbol=symbol,
            market=_MARKET,
            strategy=strategy,
            score=90.0,
            stage="breakout",
            entry=100.0,
            stop=90.0,
            target=120.0,
            features_json={},
            llm_verdict_json=(
                {
                    "symbol": symbol,
                    "verdict": verdict,
                    "confidence": 80,
                    "thesis": "t",
                    "risks": "r",
                }
                if verdict is not None
                else None
            ),
            created_at=datetime.now(UTC),
        )
        session.add(pick)
        await session.commit()
        return pick.id


async def test_paper_attribution_closed_position_uses_sell_fill_price(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    await init_db(settings)
    pick_id = await _add_pick(settings, symbol="AAPL", strategy="momentum", verdict="buy")

    now = datetime.now(UTC)
    async with async_session(settings) as session:
        session.add(
            PaperOrder(
                pick_id=pick_id,
                symbol="AAPL",
                market=_MARKET,
                side="buy",
                qty=10.0,
                order_type="next_open",
                status="filled",
                fill_price=100.0,
                slippage_bps=0.0,
                created_at=now - timedelta(days=3),
                filled_at=now - timedelta(days=2),
            )
        )
        session.add(
            PaperOrder(
                pick_id=pick_id,
                symbol="AAPL",
                market=_MARKET,
                side="sell",
                qty=10.0,
                order_type="next_open",
                status="filled",
                fill_price=120.0,
                slippage_bps=0.0,
                created_at=now - timedelta(days=1),
                filled_at=now,
            )
        )
        session.add(
            PaperPosition(
                symbol="AAPL",
                market=_MARKET,
                qty=0.0,
                avg_price=100.0,
                opened_at=now - timedelta(days=2),
                closed_at=now,
                realized_pnl=200.0,
            )
        )
        await session.commit()

    with _store(tmp_path) as store:
        rows = await paper_attribution(store=store, settings=settings)

    assert len(rows) == 1
    row = rows[0]
    assert row["symbol"] == "AAPL"
    assert row["strategy"] == "momentum"
    assert row["fill_price"] == 100.0
    assert row["exit_price"] == 120.0
    assert row["status"] == "closed"
    assert row["pnl"] == pytest.approx(200.0)
    assert row["pnl_pct"] == pytest.approx(20.0)
    assert row["llm_verdict"] == "buy"


async def test_paper_attribution_open_position_marks_to_latest_close(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    await init_db(settings)
    pick_id = await _add_pick(settings, symbol="MSFT", strategy="breakout", verdict="watch")

    now = datetime.now(UTC)
    async with async_session(settings) as session:
        session.add(
            PaperOrder(
                pick_id=pick_id,
                symbol="MSFT",
                market=_MARKET,
                side="buy",
                qty=5.0,
                order_type="next_open",
                status="filled",
                fill_price=200.0,
                slippage_bps=0.0,
                created_at=now - timedelta(days=1),
                filled_at=now,
            )
        )
        session.add(
            PaperPosition(
                symbol="MSFT", market=_MARKET, qty=5.0, avg_price=200.0, opened_at=now
            )
        )
        await session.commit()

    with _store(tmp_path) as store:
        _put_bar(store, "MSFT", now, close=210.0)
        rows = await paper_attribution(store=store, settings=settings)

    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "open"
    assert row["exit_price"] == 210.0
    assert row["pnl"] == pytest.approx((210.0 - 200.0) * 5.0)
    assert row["llm_verdict"] == "watch"


async def test_paper_attribution_ignores_orders_without_pick_or_unfilled(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    await init_db(settings)
    now = datetime.now(UTC)

    async with async_session(settings) as session:
        session.add(
            PaperOrder(
                pick_id=None,
                symbol="NOPICK",
                market=_MARKET,
                side="buy",
                qty=1.0,
                order_type="next_open",
                status="filled",
                fill_price=50.0,
                created_at=now,
                filled_at=now,
            )
        )
        session.add(
            PaperOrder(
                pick_id=None,
                symbol="PENDING",
                market=_MARKET,
                side="buy",
                qty=1.0,
                order_type="next_open",
                status="pending",
                created_at=now,
            )
        )
        await session.commit()

    with _store(tmp_path) as store:
        rows = await paper_attribution(store=store, settings=settings)

    assert rows == []


def test_attribution_summary_computes_totals_win_rate_and_breakdowns() -> None:
    rows = [
        {
            "symbol": "AAPL",
            "market": _MARKET,
            "strategy": "momentum+breakout",
            "picked_at": "2026-01-05",
            "fill_price": 100.0,
            "qty": 10.0,
            "exit_price": 120.0,
            "pnl": 200.0,
            "pnl_pct": 20.0,
            "llm_verdict": "buy",
            "status": "closed",
        },
        {
            "symbol": "MSFT",
            "market": _MARKET,
            "strategy": "momentum",
            "picked_at": "2026-01-06",
            "fill_price": 200.0,
            "qty": 5.0,
            "exit_price": 190.0,
            "pnl": -50.0,
            "pnl_pct": -5.0,
            "llm_verdict": "watch",
            "status": "closed",
        },
        {
            "symbol": "OPEN1",
            "market": _MARKET,
            "strategy": "breakout",
            "picked_at": "2026-01-07",
            "fill_price": 50.0,
            "qty": 1.0,
            "exit_price": None,
            "pnl": None,
            "pnl_pct": None,
            "llm_verdict": "buy",
            "status": "open",
        },
    ]

    summary = attribution_summary(rows)

    assert summary["total_pnl"] == pytest.approx(150.0)
    assert summary["win_rate"] == pytest.approx(0.5)  # 1 win of 2 priced rows
    assert summary["position_count"] == 3
    assert summary["priced_count"] == 2
    # AAPL's fused strategy contributes its +200 to both momentum and breakout.
    assert summary["by_strategy"]["momentum"] == pytest.approx(150.0)  # 200 - 50
    assert summary["by_strategy"]["breakout"] == pytest.approx(200.0)
    assert summary["by_verdict"]["buy"] == pytest.approx(200.0)
    assert summary["by_verdict"]["watch"] == pytest.approx(-50.0)


def test_attribution_summary_handles_empty_rows() -> None:
    summary = attribution_summary([])
    assert summary["total_pnl"] == 0.0
    assert summary["win_rate"] == 0.0
    assert summary["position_count"] == 0
    assert summary["by_strategy"] == {}
    assert summary["by_verdict"] == {}
