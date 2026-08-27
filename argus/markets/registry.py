"""Static registry of supported markets."""

from datetime import time
from zoneinfo import ZoneInfo

from argus.markets.calendars import NSE_HOLIDAYS, US_MARKET_HOLIDAYS
from argus.markets.model import Market

US_NYSE = Market(
    code="US_NYSE",
    name="New York Stock Exchange",
    timezone=ZoneInfo("America/New_York"),
    currency="USD",
    open_time=time(9, 30),
    close_time=time(16, 0),
    post_close_run=time(16, 30),
    holidays=US_MARKET_HOLIDAYS,
)

US_NASDAQ = Market(
    code="US_NASDAQ",
    name="NASDAQ",
    timezone=ZoneInfo("America/New_York"),
    currency="USD",
    open_time=time(9, 30),
    close_time=time(16, 0),
    post_close_run=time(16, 30),
    holidays=US_MARKET_HOLIDAYS,
)

IN_NSE = Market(
    code="IN_NSE",
    name="National Stock Exchange of India",
    timezone=ZoneInfo("Asia/Kolkata"),
    currency="INR",
    open_time=time(9, 15),
    close_time=time(15, 30),
    post_close_run=time(18, 30),
    holidays=NSE_HOLIDAYS,
)

_REGISTRY: dict[str, Market] = {m.code: m for m in (US_NYSE, US_NASDAQ, IN_NSE)}


def get_market(code: str) -> Market:
    try:
        return _REGISTRY[code]
    except KeyError:
        raise KeyError(f"Unknown market code: {code!r}") from None


def all_markets() -> list[Market]:
    return list(_REGISTRY.values())
