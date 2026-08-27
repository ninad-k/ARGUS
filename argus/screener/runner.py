"""``run_screen`` — end-to-end orchestration: universe -> features -> filters ->
strategies -> fusion -> ranking, plus ``persist_screen_result`` for the DB.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime

from argus.config import AppSettings
from argus.data.store.duckdb_ohlcv import BarStore
from argus.data.universe import UniverseProvider
from argus.db import async_session
from argus.db.models import DailyPick, ScreenRun
from argus.indicators.features import compute_features
from argus.markets import Instrument, Market
from argus.screener.base import Candidate, DefaultScreenContext, Strategy
from argus.screener.filters import build_default_chain
from argus.screener.registry import all_strategies

_FUSION_BONUS_PER_STRATEGY = 5.0
_MAX_SCORE = 100.0


@dataclass
class ScreenResult:
    """The outcome of one ``run_screen`` call."""

    market_code: str
    run_ts: datetime
    universe_size: int
    filtered_size: int
    candidates: list[Candidate]
    top: list[Candidate]
    rejections: dict[str, str] = field(default_factory=dict)


async def run_screen(
    market: Market,
    *,
    store: BarStore,
    universe_provider: UniverseProvider,
    strategies: list[Strategy] | None = None,
    top_n: int = 5,
) -> ScreenResult:
    """Run every applicable strategy over ``market``'s universe and rank the result.

    Pipeline: resolve the universe -> compute features per instrument from
    cached bars (skipping any instrument with no bar history) -> run the
    default filter chain -> hand survivors to each strategy that
    ``supports(market)`` -> fuse candidates picked by multiple strategies for
    the same (symbol, direction) -> rank by fused score, descending.
    """
    run_ts = datetime.now(UTC)

    if strategies is None:
        strategies = [cls() for cls in all_strategies().values()]
    applicable = [s for s in strategies if s.supports(market)]

    instruments = await universe_provider.universe(market)

    features_by_symbol: dict[str, dict[str, float]] = {}
    usable: list[Instrument] = []
    rejections: dict[str, str] = {}
    for inst in instruments:
        bars = await asyncio.to_thread(store.get_bars, inst.market_code, inst.symbol, 260)
        if len(bars) == 0:
            rejections[inst.symbol] = "no bar data available"
            continue
        features_by_symbol[inst.symbol] = compute_features(bars)
        usable.append(inst)

    chain = build_default_chain(market)
    passed, chain_rejections = chain.run(usable, features_by_symbol)
    rejections.update(chain_rejections)

    ctx = DefaultScreenContext(market, passed, store, feature_cache=features_by_symbol)

    all_candidates: list[Candidate] = []
    for strategy in applicable:
        all_candidates.extend(await strategy.screen(ctx))

    fused = _fuse(all_candidates)
    fused.sort(key=lambda c: c.score, reverse=True)

    return ScreenResult(
        market_code=market.code,
        run_ts=run_ts,
        universe_size=len(instruments),
        filtered_size=len(passed),
        candidates=fused,
        top=fused[:top_n],
        rejections=rejections,
    )


def _fuse(candidates: list[Candidate]) -> list[Candidate]:
    """Merge candidates picked by multiple strategies for the same (symbol, direction).

    Fused score = max(individual scores) + 5 * (k - 1), capped at 100, where
    ``k`` is the number of strategies that picked the instrument. Reasons are
    concatenated (tagged by strategy) and ``strategy`` becomes e.g.
    ``"momentum+breakout"``.
    """
    groups: dict[tuple[str, str], list[Candidate]] = {}
    for c in candidates:
        groups.setdefault((c.instrument.symbol, c.direction), []).append(c)

    fused: list[Candidate] = []
    for group in groups.values():
        if len(group) == 1:
            fused.append(group[0])
            continue

        best = max(group, key=lambda c: c.score)
        bonus = _FUSION_BONUS_PER_STRATEGY * (len(group) - 1)
        score = min(_MAX_SCORE, best.score + bonus)
        strategy_name = "+".join(dict.fromkeys(c.strategy for c in group))
        reason = " | ".join(f"[{c.strategy}] {c.reason}" for c in group if c.reason)
        merged_features: dict[str, float] = {}
        for c in group:
            merged_features.update(c.features)

        fused.append(
            Candidate(
                instrument=best.instrument,
                strategy=strategy_name,
                score=score,
                direction=best.direction,
                stage=best.stage,
                reason=reason,
                entry=best.entry,
                stop=best.stop,
                target=best.target,
                features=merged_features,
            )
        )
    return fused


async def persist_screen_result(
    result: ScreenResult, settings: AppSettings | None = None
) -> int:
    """Persist a ``ScreenResult`` as one ``ScreenRun`` + its ``DailyPick`` rows.

    ``features_json`` is round-tripped through ``json.dumps``/``json.loads``
    (rather than assigned the raw ``Candidate.features`` dict) so any NaN
    feature values are normalized the same way they'll come back out on
    read. The model column is typed ``dict[str, Any]`` (SQLAlchemy's
    ``JSON`` type serializes on write), so the final value handed to the ORM
    is a plain dict, not a JSON string. Returns the new run's id.
    """
    async with async_session(settings) as session:
        strategies_used = sorted(
            {name for c in result.candidates for name in c.strategy.split("+")}
        )
        run = ScreenRun(
            market=result.market_code,
            run_ts=result.run_ts,
            universe_size=result.universe_size,
            strategies_json={"strategies": strategies_used},
            status="completed",
            duration_ms=None,
        )
        session.add(run)
        await session.flush()

        for c in result.candidates:
            session.add(
                DailyPick(
                    run_id=run.id,
                    symbol=c.instrument.symbol,
                    market=c.instrument.market_code,
                    strategy=c.strategy,
                    score=c.score,
                    stage=c.stage,
                    reason=c.reason,
                    entry=c.entry,
                    stop=c.stop,
                    target=c.target,
                    features_json=json.loads(json.dumps(c.features)),
                    llm_verdict_json=None,
                    created_at=result.run_ts,
                )
            )
        await session.commit()
        return run.id
