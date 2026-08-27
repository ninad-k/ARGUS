"""CompositePriceProvider fall-through and market-support filtering."""

from dataclasses import dataclass, field
from datetime import UTC, date, datetime

import numpy as np

from argus.data.prices.base import BAR_DTYPE, ProviderHealth, Quote
from argus.data.prices.composite import CompositePriceProvider
from argus.data.prices.static_provider import StaticPriceProvider, synthetic_bars
from argus.markets import IN_NSE, US_NASDAQ, Instrument, Market


@dataclass
class _FakeProvider:
    """Minimal PriceDataProvider stand-in with controllable market support."""

    name: str
    supported_markets: set[str]
    bars: dict[str, np.ndarray] = field(default_factory=dict)
    calls: list[str] = field(default_factory=list)

    def supports(self, market: Market) -> bool:
        return market.code in self.supported_markets

    async def get_daily_bars(self, inst: Instrument, start: date, end: date) -> np.ndarray:
        self.calls.append(inst.symbol)
        return self.bars.get(inst.symbol, np.zeros(0, dtype=BAR_DTYPE))

    async def get_quote(self, inst: Instrument) -> Quote | None:
        self.calls.append(inst.symbol)
        return None

    async def health_check(self) -> ProviderHealth:
        return ProviderHealth(ok=True, detail=self.name, checked_at=datetime.now(UTC))


async def test_composite_falls_through_when_first_provider_empty() -> None:
    bars = synthetic_bars(n=5, start_price=10.0, seed=1, start=date(2026, 1, 2))
    empty_provider = _FakeProvider(name="empty", supported_markets={"US_NASDAQ"})
    second = StaticPriceProvider({"AAPL": bars})
    composite = CompositePriceProvider([empty_provider, second])

    inst = Instrument(symbol="AAPL", market_code=US_NASDAQ.code)
    result = await composite.get_daily_bars(inst, date(2026, 1, 2), date(2026, 1, 6))

    assert len(result) == 5
    assert empty_provider.calls == ["AAPL"]  # first provider was tried and skipped


async def test_composite_skips_provider_that_does_not_support_market() -> None:
    bars = synthetic_bars(n=5, start_price=10.0, seed=1, start=date(2026, 1, 2))
    us_only = _FakeProvider(
        name="us_only", supported_markets={"US_NASDAQ"}, bars={"RELIANCE": bars}
    )
    fallback = StaticPriceProvider({"RELIANCE": bars})
    composite = CompositePriceProvider([us_only, fallback])

    inst = Instrument(symbol="RELIANCE", market_code=IN_NSE.code)
    result = await composite.get_daily_bars(inst, date(2026, 1, 2), date(2026, 1, 6))

    assert len(result) == 5
    assert us_only.calls == []  # skipped entirely — doesn't support IN_NSE


async def test_composite_get_quote_falls_through() -> None:
    bars = synthetic_bars(n=5, start_price=10.0, seed=1, start=date(2026, 1, 2))
    none_provider = _FakeProvider(name="none", supported_markets={"US_NASDAQ"})
    second = StaticPriceProvider({"AAPL": bars})
    composite = CompositePriceProvider([none_provider, second])

    inst = Instrument(symbol="AAPL", market_code=US_NASDAQ.code)
    quote = await composite.get_quote(inst)

    assert quote is not None
    assert quote.symbol == "AAPL"


async def test_composite_health_check_aggregates_ok_if_any_provider_ok() -> None:
    healthy = StaticPriceProvider()
    composite = CompositePriceProvider([healthy])
    health = await composite.health_check()
    assert health.ok is True


async def test_composite_supports_reflects_underlying_providers() -> None:
    us_only = _FakeProvider(name="us_only", supported_markets={"US_NASDAQ"})
    composite = CompositePriceProvider([us_only])
    assert composite.supports(US_NASDAQ) is True
    assert composite.supports(IN_NSE) is False


@dataclass
class _CloseableFakeProvider(_FakeProvider):
    """A ``_FakeProvider`` that also owns a closeable resource, mirroring
    ``NSEProvider``'s ``httpx.AsyncClient`` -- exercises
    ``CompositePriceProvider.aclose``'s structural ``aclose`` detection."""

    closed: bool = False

    async def aclose(self) -> None:
        self.closed = True


async def test_composite_aclose_closes_members_that_have_aclose() -> None:
    closeable = _CloseableFakeProvider(name="closeable", supported_markets={"US_NASDAQ"})
    plain = _FakeProvider(name="plain", supported_markets={"US_NASDAQ"})
    composite = CompositePriceProvider([closeable, plain])

    await composite.aclose()  # must not raise for `plain`, which has no aclose

    assert closeable.closed is True
