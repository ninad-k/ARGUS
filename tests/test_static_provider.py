"""synthetic_bars determinism and StaticPriceProvider behavior."""

from datetime import date

import numpy as np

from argus.data.prices.static_provider import StaticPriceProvider, synthetic_bars
from argus.markets import Instrument


def test_synthetic_bars_same_seed_is_identical() -> None:
    a = synthetic_bars(n=30, start_price=100.0, seed=42, start=date(2026, 1, 2))
    b = synthetic_bars(n=30, start_price=100.0, seed=42, start=date(2026, 1, 2))
    assert np.array_equal(a, b)


def test_synthetic_bars_different_seed_differs() -> None:
    a = synthetic_bars(n=30, start_price=100.0, seed=1, start=date(2026, 1, 2))
    b = synthetic_bars(n=30, start_price=100.0, seed=2, start=date(2026, 1, 2))
    assert not np.array_equal(a, b)


def test_synthetic_bars_shape_and_positivity() -> None:
    bars = synthetic_bars(n=15, start_price=50.0, seed=7, start=date(2026, 1, 2))
    assert len(bars) == 15
    assert np.all(bars["high"] >= bars["low"])
    assert np.all(bars["close"] > 0)
    assert np.all(bars["volume"] >= 0)


async def test_static_provider_returns_bars_for_range() -> None:
    bars = synthetic_bars(n=10, start_price=100.0, seed=5, start=date(2026, 1, 2))
    provider = StaticPriceProvider({"AAPL": bars})
    inst = Instrument(symbol="AAPL", market_code="US_NASDAQ")

    result = await provider.get_daily_bars(inst, date(2026, 1, 2), date(2026, 1, 11))
    assert len(result) == 10
    assert np.array_equal(result, bars)


async def test_static_provider_missing_symbol_returns_empty() -> None:
    provider = StaticPriceProvider()
    inst = Instrument(symbol="NOPE", market_code="US_NASDAQ")
    result = await provider.get_daily_bars(inst, date(2026, 1, 2), date(2026, 1, 11))
    assert len(result) == 0


async def test_static_provider_get_quote_uses_last_bar() -> None:
    bars = synthetic_bars(n=5, start_price=100.0, seed=9, start=date(2026, 1, 2))
    provider = StaticPriceProvider({"AAPL": bars})
    inst = Instrument(symbol="AAPL", market_code="US_NASDAQ")

    quote = await provider.get_quote(inst)
    assert quote is not None
    assert quote.price == float(bars[-1]["close"])
    assert quote.prev_close == float(bars[-2]["close"])


async def test_static_provider_health_check_always_ok() -> None:
    provider = StaticPriceProvider()
    health = await provider.health_check()
    assert health.ok is True
