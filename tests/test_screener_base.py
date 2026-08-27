"""Strategy registry discovery/gating, and DefaultScreenContext memoization."""

from datetime import date
from pathlib import Path

from argus.data.prices.static_provider import synthetic_bars
from argus.data.store.duckdb_ohlcv import BarStore
from argus.markets import IN_NSE, US_NASDAQ, Instrument
from argus.screener.base import Candidate, DefaultScreenContext, ScreenContext, Strategy
from argus.screener.registry import all_strategies, get_strategy


class _AllMarketsStrategy(Strategy):
    slug = "_test_all_markets"
    name = "test-all-markets"

    async def screen(self, ctx: ScreenContext) -> list[Candidate]:
        return []


class _UsOnlyStrategy(Strategy):
    slug = "_test_us_only"
    name = "test-us-only"
    markets = frozenset({US_NASDAQ.code})

    async def screen(self, ctx: ScreenContext) -> list[Candidate]:
        return []


def test_registry_discovers_builtin_strategies() -> None:
    strategies = all_strategies()
    assert "momentum" in strategies
    assert "breakout" in strategies


def test_get_strategy_returns_registered_class() -> None:
    cls = get_strategy("momentum")
    assert cls.slug == "momentum"


def test_get_strategy_unknown_slug_raises_keyerror() -> None:
    try:
        get_strategy("does-not-exist")
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError")


def test_supports_defaults_to_all_markets() -> None:
    strategy = _AllMarketsStrategy()
    assert strategy.supports(US_NASDAQ)
    assert strategy.supports(IN_NSE)


def test_supports_restricts_to_declared_markets() -> None:
    strategy = _UsOnlyStrategy()
    assert strategy.supports(US_NASDAQ)
    assert not strategy.supports(IN_NSE)


def test_builtin_strategies_support_all_markets() -> None:
    for cls in all_strategies().values():
        strategy = cls()
        assert strategy.supports(US_NASDAQ)
        assert strategy.supports(IN_NSE)


async def test_default_screen_context_universe_returns_fixed_instruments(tmp_path: Path) -> None:
    inst = Instrument(symbol="AAPL", market_code=US_NASDAQ.code)
    with BarStore(tmp_path / "bars.duckdb") as store:
        ctx = DefaultScreenContext(US_NASDAQ, [inst], store)
        universe = await ctx.universe()
    assert universe == [inst]


async def test_default_screen_context_bars_reads_through_to_store(tmp_path: Path) -> None:
    bars = synthetic_bars(n=30, start_price=100.0, seed=1, start=date(2026, 1, 2))
    inst = Instrument(symbol="AAPL", market_code=US_NASDAQ.code)
    with BarStore(tmp_path / "bars.duckdb") as store:
        store.upsert_bars(US_NASDAQ.code, "AAPL", bars)
        ctx = DefaultScreenContext(US_NASDAQ, [inst], store)

        fetched = await ctx.bars(inst, n=10)
    assert len(fetched) == 10
    assert list(fetched["ts"]) == list(bars["ts"][-10:])


async def test_default_screen_context_features_are_memoized(tmp_path: Path) -> None:
    bars = synthetic_bars(n=260, start_price=100.0, seed=2, start=date(2026, 1, 2))
    inst = Instrument(symbol="AAPL", market_code=US_NASDAQ.code)
    with BarStore(tmp_path / "bars.duckdb") as store:
        store.upsert_bars(US_NASDAQ.code, "AAPL", bars)
        ctx = DefaultScreenContext(US_NASDAQ, [inst], store)

        first = await ctx.features(inst)
        second = await ctx.features(inst)

    # Same dict object returned both times -- proof compute_features() only
    # ran once and the second call was served from the cache.
    assert first is second


async def test_default_screen_context_features_prepopulated_cache_is_used(
    tmp_path: Path,
) -> None:
    inst = Instrument(symbol="AAPL", market_code=US_NASDAQ.code)
    sentinel = {"close": 123.45}
    with BarStore(tmp_path / "bars.duckdb") as store:
        # No bars were ever upserted for this symbol -- if the cache weren't
        # honored, features() would hit the (empty) store and return NaNs.
        ctx = DefaultScreenContext(
            US_NASDAQ, [inst], store, feature_cache={"AAPL": sentinel}
        )
        result = await ctx.features(inst)
    assert result is sentinel
