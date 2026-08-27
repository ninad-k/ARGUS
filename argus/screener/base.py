"""Strategy ABC, ``Candidate`` DTO, and ``ScreenContext`` — the unified screener core.

DRUVA kept "Strategy" (per-symbol signal generation) and "Scanner" (universe-wide
screening) as two separate abstractions. ARGUS deliberately merges them: a single
``Strategy`` runs across the whole (already-filtered) universe and emits ranked
``Candidate``s directly — there is no separate per-symbol signal layer sitting
underneath it.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from argus.data.store.duckdb_ohlcv import BarStore
from argus.indicators.features import compute_features
from argus.markets import Instrument, Market

if TYPE_CHECKING:
    # Deferred to avoid a runtime cycle: argus.advisor.pick_reviewer imports
    # Candidate from this module. Safe under `from __future__ import
    # annotations` — the annotation below is never evaluated at runtime.
    from argus.advisor.pick_reviewer import PickVerdict


@dataclass
class Candidate:
    """A single screener hit: one instrument, one strategy, one directional call."""

    instrument: Instrument
    strategy: str
    score: float  # 0-100
    direction: Literal["long", "short"] = "long"
    stage: str = ""
    reason: str = ""
    entry: float | None = None
    stop: float | None = None
    target: float | None = None
    features: dict[str, float] = field(default_factory=dict)
    llm_verdict: PickVerdict | None = None


class ScreenContext(Protocol):
    """What the runner exposes to a ``Strategy`` at execution time."""

    market: Market

    async def universe(self) -> list[Instrument]:
        """The (already filtered) instruments this run is screening."""
        ...

    async def bars(self, inst: Instrument, n: int = 260) -> NDArray[np.void]:
        """The most recent ``n`` daily bars for ``inst``, ascending by ts."""
        ...

    async def features(self, inst: Instrument) -> dict[str, float]:
        """``compute_features`` output for ``inst``, memoized per symbol."""
        ...


class Strategy(ABC):
    """Base class for a screener strategy.

    Subclasses set ``slug``/``name`` as class attributes and implement
    ``screen()``. ``markets`` restricts which markets the strategy applies to
    (``None`` means "all markets").
    """

    slug: str
    name: str
    markets: frozenset[str] | None = None

    def supports(self, market: Market) -> bool:
        return self.markets is None or market.code in self.markets

    @abstractmethod
    async def screen(self, ctx: ScreenContext) -> list[Candidate]:
        """Run this strategy over ``ctx``'s universe and return candidates."""


class DefaultScreenContext:
    """Concrete ``ScreenContext`` backed by a ``BarStore`` and a fixed instrument list.

    ``bars()`` reads through to the (synchronous) ``BarStore`` via
    ``asyncio.to_thread``. ``features()`` memoizes per symbol so strategies
    that both call it (or a runner that pre-populates the cache from its own
    filter pass) never recompute ``compute_features`` twice for the same
    instrument within a run.
    """

    def __init__(
        self,
        market: Market,
        instruments: list[Instrument],
        store: BarStore,
        feature_cache: dict[str, dict[str, float]] | None = None,
    ) -> None:
        self.market = market
        self._instruments = instruments
        self._store = store
        self._feature_cache: dict[str, dict[str, float]] = (
            feature_cache if feature_cache is not None else {}
        )

    async def universe(self) -> list[Instrument]:
        return list(self._instruments)

    async def bars(self, inst: Instrument, n: int = 260) -> NDArray[np.void]:
        return await asyncio.to_thread(
            self._store.get_bars, inst.market_code, inst.symbol, n
        )

    async def features(self, inst: Instrument) -> dict[str, float]:
        cached = self._feature_cache.get(inst.symbol)
        if cached is not None:
            return cached
        bars = await self.bars(inst)
        computed = compute_features(bars)
        self._feature_cache[inst.symbol] = computed
        return computed
