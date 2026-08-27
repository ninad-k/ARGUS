"""Pick outcome evaluation: did each daily pick hit its target, its stop,
neither, or run past its evaluation horizon?

Purely a read over ``DailyPick`` rows + ``BarStore`` bars -- nothing here
ever touches a pick or a bar. Every entry point is exception-contained: a
malformed row (bad prices, missing bars) is skipped and logged rather than
raised, since this walks historical data whose shape this module doesn't
control.
"""

from __future__ import annotations

import asyncio
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Literal

import numpy as np
import structlog
from sqlalchemy import select

from argus.config import AppSettings, get_settings
from argus.data.store.duckdb_ohlcv import BarStore
from argus.db import async_session
from argus.db.models import DailyPick, ScreenRun

logger = structlog.get_logger(__name__)

Status = Literal["hit_target", "hit_stop", "open", "expired"]
_STATUSES: tuple[Status, ...] = ("hit_target", "hit_stop", "open", "expired")

_DEFAULT_HORIZON_DAYS = 30
_DEFAULT_LIMIT_RUNS = 30


@dataclass(frozen=True)
class PickOutcome:
    """One pick's realized (or still-open) outcome as of the latest bar seen."""

    pick_id: int
    symbol: str
    market: str
    strategy: str
    picked_at: date
    entry: float
    stop: float | None
    target: float | None
    days_held: int
    status: Status
    return_pct: float | None
    max_favorable_pct: float
    max_adverse_pct: float


def _bar_date(bar: np.void) -> date:
    """The calendar date of a ``BAR_DTYPE`` row's ``ts`` field."""
    ts: datetime = bar["ts"].astype("datetime64[s]").astype(datetime)
    return ts.date()


async def evaluate_pick(
    pick: DailyPick, store: BarStore, *, horizon_days: int = _DEFAULT_HORIZON_DAYS
) -> PickOutcome | None:
    """Walk bars after ``pick``'s date forward and classify its outcome.

    Returns ``None`` when there's no usable entry price, or no bars exist
    strictly after the pick's date yet -- both mean this pick can't be
    evaluated at all.

    Within each bar, the stop is checked *before* the target (a
    conservative assumption: OHLC bars can't tell us which level was
    touched first intraday, so a bar that round-trips through both is
    scored as a stop-out rather than a win). The walk stops at the first
    bar that breaches either level, or after ``horizon_days`` bars with
    neither breached (``"expired"``), or runs out of bars first (still
    ``"open"``). ``return_pct`` is the latest evaluated close vs. entry.
    """
    if pick.entry is None or pick.entry <= 0:
        return None

    picked_date = pick.created_at.date()
    all_bars = await asyncio.to_thread(store.get_bars, pick.market, pick.symbol, None)
    after = [b for b in all_bars if _bar_date(b) > picked_date]
    if not after:
        return None

    entry = pick.entry
    max_fav = 0.0
    max_adv = 0.0
    status: Status = "open"
    days_held = 0
    last_close = entry

    window = after[:horizon_days]
    for i, bar in enumerate(window):
        days_held = i + 1
        high = float(bar["high"])
        low = float(bar["low"])
        last_close = float(bar["close"])

        max_fav = max(max_fav, (high - entry) / entry * 100)
        max_adv = max(max_adv, (entry - low) / entry * 100)

        if pick.stop is not None and low <= pick.stop:
            status = "hit_stop"
            break
        if pick.target is not None and high >= pick.target:
            status = "hit_target"
            break
    else:
        status = "expired" if len(after) >= horizon_days else "open"

    return PickOutcome(
        pick_id=pick.id,
        symbol=pick.symbol,
        market=pick.market,
        strategy=pick.strategy,
        picked_at=picked_date,
        entry=entry,
        stop=pick.stop,
        target=pick.target,
        days_held=days_held,
        status=status,
        return_pct=round((last_close - entry) / entry * 100, 2),
        max_favorable_pct=round(max_fav, 2),
        max_adverse_pct=round(max_adv, 2),
    )


async def evaluate_run_history(
    market_code: str | None,
    store: BarStore,
    *,
    limit_runs: int = _DEFAULT_LIMIT_RUNS,
    horizon_days: int = _DEFAULT_HORIZON_DAYS,
    settings: AppSettings | None = None,
) -> list[PickOutcome]:
    """Evaluate every pick from the ``limit_runs`` most recent screen runs.

    ``market_code=None`` considers runs across every market. A per-pick
    evaluation failure is logged and skipped rather than aborting the whole
    batch -- one bad row must never hide every other pick's outcome.
    """
    settings = settings or get_settings()

    async with async_session(settings) as session:
        stmt = select(ScreenRun).order_by(ScreenRun.run_ts.desc(), ScreenRun.id.desc())
        if market_code is not None:
            stmt = stmt.where(ScreenRun.market == market_code)
        runs = (await session.execute(stmt.limit(limit_runs))).scalars().all()
        if not runs:
            return []

        run_ids = [r.id for r in runs]
        picks = (
            await session.execute(select(DailyPick).where(DailyPick.run_id.in_(run_ids)))
        ).scalars().all()

    outcomes: list[PickOutcome] = []
    for pick in picks:
        try:
            outcome = await evaluate_pick(pick, store, horizon_days=horizon_days)
        except Exception as exc:  # a bad row must never abort the batch
            logger.warning("analysis.outcomes.evaluate_failed", pick_id=pick.id, error=str(exc))
            continue
        if outcome is not None:
            outcomes.append(outcome)
    return outcomes


def _stats_for(outcomes: Sequence[PickOutcome]) -> dict[str, Any]:
    decided = [o for o in outcomes if o.status in ("hit_target", "hit_stop")]
    n_decided = len(decided)
    n_targets = sum(1 for o in decided if o.status == "hit_target")
    n_stops = sum(1 for o in decided if o.status == "hit_stop")

    hit_rate = round(n_targets / n_decided, 4) if n_decided else 0.0
    stop_rate = round(n_stops / n_decided, 4) if n_decided else 0.0

    returns = [o.return_pct for o in outcomes if o.return_pct is not None]
    avg_return = round(sum(returns) / len(returns), 2) if returns else 0.0

    winners = [
        o.return_pct for o in outcomes if o.status == "hit_target" and o.return_pct is not None
    ]
    losers = [
        o.return_pct for o in outcomes if o.status == "hit_stop" and o.return_pct is not None
    ]
    avg_winner = round(sum(winners) / len(winners), 2) if winners else 0.0
    avg_loser = round(sum(losers) / len(losers), 2) if losers else 0.0

    expectancy = round(hit_rate * avg_winner + stop_rate * avg_loser, 2)

    counts = Counter(o.status for o in outcomes)

    return {
        "total": len(outcomes),
        "hit_rate": hit_rate,
        "stop_rate": stop_rate,
        "avg_return_pct": avg_return,
        "avg_winner_pct": avg_winner,
        "avg_loser_pct": avg_loser,
        "expectancy": expectancy,
        "counts": {status: counts.get(status, 0) for status in _STATUSES},
    }


def summarize_outcomes(outcomes: Sequence[PickOutcome]) -> dict[str, Any]:
    """Aggregate stats over ``outcomes``, plus a per-strategy breakdown.

    ``strategy`` may be a fused label like ``"momentum+breakout"`` (see
    ``argus.screener.runner._fuse``) -- each outcome is attributed to every
    component strategy in that label, not just the first, so a pick that
    both momentum and breakout picked counts toward both strategies' stats.
    """
    overall = _stats_for(outcomes)

    by_strategy: dict[str, list[PickOutcome]] = defaultdict(list)
    for o in outcomes:
        for strat in o.strategy.split("+"):
            if strat:
                by_strategy[strat].append(o)

    overall["by_strategy"] = {
        strat: _stats_for(group) for strat, group in sorted(by_strategy.items())
    }
    return overall
