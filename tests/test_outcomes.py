"""``evaluate_pick``'s target/stop/expired/open classification and
``summarize_outcomes``'s aggregate + per-strategy math, including fused
("momentum+breakout") strategy attribution.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from argus.analysis.outcomes import (
    PickOutcome,
    evaluate_pick,
    evaluate_run_history,
    summarize_outcomes,
)
from argus.config import AppSettings
from argus.data.prices.base import bars_from_columns
from argus.data.store.duckdb_ohlcv import BarStore
from argus.db import async_session, init_db
from argus.db.models import DailyPick, ScreenRun

_MARKET = "US_NASDAQ"
_PICKED_DATE = date(2026, 1, 5)


def _settings(tmp_path: Path) -> AppSettings:
    return AppSettings(data_dir=tmp_path, _env_file=None)  # type: ignore[call-arg]


def _store(tmp_path: Path) -> BarStore:
    return BarStore(tmp_path / "market_data.duckdb")


def _put_bars(
    store: BarStore,
    symbol: str,
    start: date,
    ohlc: list[tuple[float, float, float, float]],
) -> None:
    """Upsert one bar per day starting at ``start``, each ``(open, high, low, close)``."""
    n = len(ohlc)
    ts = np.array(
        [np.datetime64((start + timedelta(days=i)).isoformat(), "s") for i in range(n)],
        dtype="datetime64[s]",
    )
    bars = bars_from_columns(
        ts,
        np.array([o for o, _, _, _ in ohlc]),
        np.array([h for _, h, _, _ in ohlc]),
        np.array([l for _, _, l, _ in ohlc]),  # noqa: E741 -- matches the tuple's own name
        np.array([c for _, _, _, c in ohlc]),
        np.array([1_000_000.0] * n),
    )
    store.upsert_bars(_MARKET, symbol, bars)


def _pick(
    *,
    id: int = 1,  # noqa: A002 -- matches DailyPick's own column name
    symbol: str = "AAPL",
    strategy: str = "momentum",
    entry: float = 100.0,
    stop: float | None = 90.0,
    target: float | None = 110.0,
    picked_at: date = _PICKED_DATE,
) -> DailyPick:
    pick = DailyPick(
        run_id=1,
        symbol=symbol,
        market=_MARKET,
        strategy=strategy,
        score=90.0,
        stage="breakout",
        entry=entry,
        stop=stop,
        target=target,
        features_json={},
        created_at=datetime.combine(picked_at, datetime.min.time(), tzinfo=UTC),
    )
    pick.id = id  # DailyPick.id is normally DB-assigned; tests build rows in-memory
    return pick


async def test_evaluate_pick_returns_none_without_entry(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        outcome = await evaluate_pick(_pick(entry=0.0), store)
    assert outcome is None


async def test_evaluate_pick_returns_none_when_no_bars_after_pick_date(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        _put_bars(store, "AAPL", _PICKED_DATE - timedelta(days=1), [(99, 101, 98, 100)])
        outcome = await evaluate_pick(_pick(), store)
    assert outcome is None


async def test_evaluate_pick_hits_target(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        _put_bars(
            store,
            "AAPL",
            _PICKED_DATE + timedelta(days=1),
            [
                (101, 105, 99, 103),  # day 1: no breach
                (104, 112, 103, 111),  # day 2: high 112 >= target 110
            ],
        )
        outcome = await evaluate_pick(_pick(), store)

    assert outcome is not None
    assert outcome.status == "hit_target"
    assert outcome.days_held == 2
    assert outcome.return_pct == pytest.approx((111 - 100) / 100 * 100, abs=0.01)
    assert outcome.max_favorable_pct == pytest.approx((112 - 100) / 100 * 100, abs=0.01)


async def test_evaluate_pick_hits_stop(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        _put_bars(
            store,
            "AAPL",
            _PICKED_DATE + timedelta(days=1),
            [(99, 101, 85, 86)],  # low 85 <= stop 90
        )
        outcome = await evaluate_pick(_pick(), store)

    assert outcome is not None
    assert outcome.status == "hit_stop"
    assert outcome.days_held == 1
    assert outcome.max_adverse_pct == pytest.approx((100 - 85) / 100 * 100, abs=0.01)


async def test_evaluate_pick_same_bar_stop_wins_over_target(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        _put_bars(
            store,
            "AAPL",
            _PICKED_DATE + timedelta(days=1),
            [(100, 115, 85, 90)],  # both stop (low 85) and target (high 115) breached
        )
        outcome = await evaluate_pick(_pick(), store)

    assert outcome is not None
    assert outcome.status == "hit_stop"


async def test_evaluate_pick_expires_after_horizon(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        _put_bars(
            store,
            "AAPL",
            _PICKED_DATE + timedelta(days=1),
            [(100, 102, 98, 101)] * 3,  # never breaches either level
        )
        outcome = await evaluate_pick(_pick(), store, horizon_days=3)

    assert outcome is not None
    assert outcome.status == "expired"
    assert outcome.days_held == 3


async def test_evaluate_pick_stays_open_within_horizon(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        _put_bars(
            store,
            "AAPL",
            _PICKED_DATE + timedelta(days=1),
            [(100, 102, 98, 101)],  # only 1 bar so far, horizon is 30
        )
        outcome = await evaluate_pick(_pick(), store, horizon_days=30)

    assert outcome is not None
    assert outcome.status == "open"
    assert outcome.days_held == 1


async def test_evaluate_pick_no_stop_or_target_never_hits(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        _put_bars(
            store,
            "AAPL",
            _PICKED_DATE + timedelta(days=1),
            [(100, 999, 1, 101)],  # would breach any level, but none is set
        )
        outcome = await evaluate_pick(_pick(stop=None, target=None), store, horizon_days=30)

    assert outcome is not None
    assert outcome.status == "open"


def _make_outcome(
    *,
    symbol: str = "AAPL",
    strategy: str = "momentum",
    status: str = "hit_target",
    return_pct: float = 5.0,
) -> PickOutcome:
    return PickOutcome(
        pick_id=1,
        symbol=symbol,
        market=_MARKET,
        strategy=strategy,
        picked_at=_PICKED_DATE,
        entry=100.0,
        stop=90.0,
        target=110.0,
        days_held=2,
        status=status,  # type: ignore[arg-type]
        return_pct=return_pct,
        max_favorable_pct=abs(return_pct) + 1,
        max_adverse_pct=1.0,
    )


def test_summarize_outcomes_computes_rates_and_expectancy() -> None:
    outcomes = [
        _make_outcome(status="hit_target", return_pct=10.0),
        _make_outcome(status="hit_target", return_pct=6.0),
        _make_outcome(status="hit_stop", return_pct=-4.0),
        _make_outcome(status="open", return_pct=1.0),
    ]

    summary = summarize_outcomes(outcomes)

    # decided = 3 (2 targets + 1 stop); hit_rate = 2/3, stop_rate = 1/3
    assert summary["hit_rate"] == pytest.approx(2 / 3, abs=1e-4)
    assert summary["stop_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert summary["avg_winner_pct"] == pytest.approx(8.0, abs=0.01)
    assert summary["avg_loser_pct"] == pytest.approx(-4.0, abs=0.01)
    expected_expectancy = (2 / 3) * 8.0 + (1 / 3) * -4.0
    assert summary["expectancy"] == pytest.approx(expected_expectancy, abs=0.01)
    assert summary["counts"]["hit_target"] == 2
    assert summary["counts"]["hit_stop"] == 1
    assert summary["counts"]["open"] == 1
    assert summary["counts"]["expired"] == 0
    assert summary["total"] == 4


def test_summarize_outcomes_splits_fused_strategy_attribution() -> None:
    outcomes = [
        _make_outcome(strategy="momentum+breakout", status="hit_target", return_pct=10.0),
        _make_outcome(strategy="momentum", status="hit_stop", return_pct=-5.0),
        _make_outcome(strategy="breakout", status="hit_target", return_pct=8.0),
    ]

    summary = summarize_outcomes(outcomes)
    by_strategy = summary["by_strategy"]

    # The fused pick counts toward *both* momentum and breakout.
    assert set(by_strategy) == {"momentum", "breakout"}
    assert by_strategy["momentum"]["total"] == 2  # fused pick + the standalone stop-out
    assert by_strategy["breakout"]["total"] == 2  # fused pick + the standalone target-hit
    assert by_strategy["momentum"]["counts"]["hit_target"] == 1
    assert by_strategy["momentum"]["counts"]["hit_stop"] == 1
    assert by_strategy["breakout"]["counts"]["hit_target"] == 2


async def test_evaluate_run_history_bounded_by_limit_runs_and_market(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    await init_db(settings)

    async with async_session(settings) as session:
        run_a = ScreenRun(
            market=_MARKET,
            run_ts=datetime.now(UTC),
            universe_size=1,
            strategies_json={},
            status="completed",
        )
        run_b = ScreenRun(
            market="IN_NSE",
            run_ts=datetime.now(UTC),
            universe_size=1,
            strategies_json={},
            status="completed",
        )
        session.add_all([run_a, run_b])
        await session.flush()

        session.add(
            DailyPick(
                run_id=run_a.id,
                symbol="AAPL",
                market=_MARKET,
                strategy="momentum",
                score=90.0,
                stage="breakout",
                entry=100.0,
                stop=90.0,
                target=110.0,
                features_json={},
                created_at=datetime.combine(_PICKED_DATE, datetime.min.time(), tzinfo=UTC),
            )
        )
        session.add(
            DailyPick(
                run_id=run_b.id,
                symbol="RELI",
                market="IN_NSE",
                strategy="momentum",
                score=90.0,
                stage="breakout",
                entry=100.0,
                stop=90.0,
                target=110.0,
                features_json={},
                created_at=datetime.combine(_PICKED_DATE, datetime.min.time(), tzinfo=UTC),
            )
        )
        await session.commit()

    with _store(tmp_path) as store:
        _put_bars(
            store, "AAPL", _PICKED_DATE + timedelta(days=1), [(101, 112, 99, 111)]
        )
        outcomes = await evaluate_run_history(_MARKET, store, settings=settings)

    assert len(outcomes) == 1
    assert outcomes[0].symbol == "AAPL"


async def test_evaluate_run_history_skips_bad_pick_without_aborting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    await init_db(settings)

    async with async_session(settings) as session:
        run = ScreenRun(
            market=_MARKET,
            run_ts=datetime.now(UTC),
            universe_size=2,
            strategies_json={},
            status="completed",
        )
        session.add(run)
        await session.flush()
        session.add(
            DailyPick(
                run_id=run.id,
                symbol="BAD",
                market=_MARKET,
                strategy="momentum",
                score=90.0,
                stage="breakout",
                entry=100.0,
                stop=90.0,
                target=110.0,
                features_json={},
                created_at=datetime.combine(_PICKED_DATE, datetime.min.time(), tzinfo=UTC),
            )
        )
        session.add(
            DailyPick(
                run_id=run.id,
                symbol="GOOD",
                market=_MARKET,
                strategy="momentum",
                score=90.0,
                stage="breakout",
                entry=100.0,
                stop=90.0,
                target=110.0,
                features_json={},
                created_at=datetime.combine(_PICKED_DATE, datetime.min.time(), tzinfo=UTC),
            )
        )
        await session.commit()

    async def _raise_for_bad(pick: DailyPick, store: BarStore, **kwargs: object) -> None:
        if pick.symbol == "BAD":
            raise RuntimeError("boom")
        return

    monkeypatch.setattr("argus.analysis.outcomes.evaluate_pick", _raise_for_bad)

    with _store(tmp_path) as store:
        outcomes = await evaluate_run_history(_MARKET, store, settings=settings)

    assert outcomes == []
