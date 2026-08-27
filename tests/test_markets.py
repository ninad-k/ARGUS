"""Trading-day/calendar logic for the market registry."""

from datetime import date

from argus.markets import IN_NSE, US_NYSE, get_market


def test_weekend_is_not_a_trading_day() -> None:
    saturday = date(2026, 8, 22)
    sunday = date(2026, 8, 23)
    assert not US_NYSE.is_trading_day(saturday)
    assert not US_NYSE.is_trading_day(sunday)


def test_us_holiday_is_not_a_trading_day() -> None:
    christmas_2025 = date(2025, 12, 25)
    assert not US_NYSE.is_trading_day(christmas_2025)


def test_nse_holiday_is_not_a_trading_day() -> None:
    republic_day_2026 = date(2026, 1, 26)
    assert not IN_NSE.is_trading_day(republic_day_2026)


def test_a_normal_weekday_is_a_trading_day() -> None:
    a_tuesday = date(2026, 8, 25)
    assert US_NYSE.is_trading_day(a_tuesday)
    assert IN_NSE.is_trading_day(a_tuesday)


def test_next_trading_day_skips_weekend() -> None:
    friday = date(2026, 8, 21)
    assert US_NYSE.next_trading_day(friday) == date(2026, 8, 24)


def test_last_trading_day_skips_weekend() -> None:
    monday = date(2026, 8, 24)
    assert US_NYSE.last_trading_day(monday) == date(2026, 8, 21)


def test_next_trading_day_skips_holiday_and_weekend() -> None:
    # Dec 24 2025 (Wed) is a trading day; Dec 25 (Thu) is Christmas; the next
    # trading day is Dec 26 (Fri).
    assert US_NYSE.next_trading_day(date(2025, 12, 24)) == date(2025, 12, 26)


def test_get_market_returns_registered_market() -> None:
    assert get_market("US_NYSE") is US_NYSE


def test_get_market_raises_for_unknown_code() -> None:
    try:
        get_market("NOT_A_MARKET")
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError for unknown market code")
