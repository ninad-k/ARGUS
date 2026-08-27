"""Live network smoke test for YFinanceProvider. Skipped by default (see pytest addopts)."""

from datetime import date, timedelta

import pytest

from argus.data.prices.yfinance_provider import YFinanceProvider
from argus.markets import Instrument


@pytest.mark.network
async def test_yfinance_get_daily_bars_aapl_30_days() -> None:
    provider = YFinanceProvider()
    inst = Instrument(symbol="AAPL", market_code="US_NASDAQ")
    end = date.today()  # noqa: DTZ011
    start = end - timedelta(days=30)

    bars = await provider.get_daily_bars(inst, start, end)

    assert len(bars) > 0
    assert bars["close"][-1] > 0
