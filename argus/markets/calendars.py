"""Static holiday calendars for 2025-2027.

Dates are sourced from official/published exchange holiday circulars:
NYSE Group's 2025/2026/2027 holiday announcement for US markets, and the
NSE India annual trading-holiday circulars for NSE. 2027 NSE dates for
lunar-calendar festivals (Holi, Ram Navami, Ganesh Chaturthi, Diwali, Guru
Nanak Jayanti, etc.) are best-effort estimates from published Hindu
festival calendars as of mid-2026 and MUST be confirmed against the
official NSE circular once it is published (typically ~December prior
year).
"""

from datetime import date

# --- US markets (NYSE and NASDAQ share one holiday calendar) ---------------

US_MARKET_HOLIDAYS_2025: frozenset[date] = frozenset(
    {
        date(2025, 1, 1),  # New Year's Day
        date(2025, 1, 20),  # Martin Luther King Jr. Day
        date(2025, 2, 17),  # Washington's Birthday (Presidents Day)
        date(2025, 4, 18),  # Good Friday
        date(2025, 5, 26),  # Memorial Day
        date(2025, 6, 19),  # Juneteenth National Independence Day
        date(2025, 7, 4),  # Independence Day
        date(2025, 9, 1),  # Labor Day
        date(2025, 11, 27),  # Thanksgiving Day
        date(2025, 12, 25),  # Christmas Day
    }
)

US_MARKET_HOLIDAYS_2026: frozenset[date] = frozenset(
    {
        date(2026, 1, 1),  # New Year's Day
        date(2026, 1, 19),  # Martin Luther King Jr. Day
        date(2026, 2, 16),  # Washington's Birthday (Presidents Day)
        date(2026, 4, 3),  # Good Friday
        date(2026, 5, 25),  # Memorial Day
        date(2026, 6, 19),  # Juneteenth National Independence Day
        date(2026, 7, 3),  # Independence Day (observed; Jul 4 falls on Saturday)
        date(2026, 9, 7),  # Labor Day
        date(2026, 11, 26),  # Thanksgiving Day
        date(2026, 12, 25),  # Christmas Day
    }
)

US_MARKET_HOLIDAYS_2027: frozenset[date] = frozenset(
    {
        date(2027, 1, 1),  # New Year's Day
        date(2027, 1, 18),  # Martin Luther King Jr. Day
        date(2027, 2, 15),  # Washington's Birthday (Presidents Day)
        date(2027, 3, 26),  # Good Friday
        date(2027, 5, 31),  # Memorial Day
        date(2027, 6, 18),  # Juneteenth National Independence Day (observed; Jun 19 falls on Sat)
        date(2027, 7, 5),  # Independence Day (observed; Jul 4 falls on Sunday)
        date(2027, 9, 6),  # Labor Day
        date(2027, 11, 25),  # Thanksgiving Day
        date(2027, 12, 24),  # Christmas Day (observed; Dec 25 falls on Saturday)
    }
)

US_MARKET_HOLIDAYS: frozenset[date] = (
    US_MARKET_HOLIDAYS_2025 | US_MARKET_HOLIDAYS_2026 | US_MARKET_HOLIDAYS_2027
)

# --- NSE India ---------------------------------------------------------------

NSE_HOLIDAYS_2025: frozenset[date] = frozenset(
    {
        date(2025, 2, 26),  # Mahashivratri
        date(2025, 3, 14),  # Holi
        date(2025, 3, 31),  # Id-Ul-Fitr (Ramzan Id)
        date(2025, 4, 10),  # Shri Mahavir Jayanti
        date(2025, 4, 14),  # Dr. Baba Saheb Ambedkar Jayanti
        date(2025, 4, 18),  # Good Friday
        date(2025, 5, 1),  # Maharashtra Day
        date(2025, 8, 15),  # Independence Day
        date(2025, 8, 27),  # Ganesh Chaturthi
        date(2025, 10, 2),  # Mahatma Gandhi Jayanti / Dussehra
        date(2025, 10, 21),  # Diwali Laxmi Pujan
        date(2025, 10, 22),  # Diwali-Balipratipada
        date(2025, 11, 5),  # Prakash Gurpurb Sri Guru Nanak Dev
        date(2025, 12, 25),  # Christmas
    }
)

NSE_HOLIDAYS_2026: frozenset[date] = frozenset(
    {
        date(2026, 1, 15),  # Special holiday: Maharashtra municipal elections
        date(2026, 1, 26),  # Republic Day
        date(2026, 3, 3),  # Holi
        date(2026, 3, 26),  # Shri Ram Navami
        date(2026, 3, 31),  # Shri Mahavir Jayanti
        date(2026, 4, 3),  # Good Friday
        date(2026, 4, 14),  # Dr. Baba Saheb Ambedkar Jayanti
        date(2026, 5, 1),  # Maharashtra Day
        date(2026, 5, 28),  # Bakri Id
        date(2026, 6, 26),  # Muharram
        date(2026, 9, 14),  # Ganesh Chaturthi
        date(2026, 10, 2),  # Mahatma Gandhi Jayanti
        date(2026, 10, 20),  # Dussehra
        date(2026, 11, 10),  # Diwali-Balipratipada
        date(2026, 11, 24),  # Prakash Gurpurb Sri Guru Nanak Dev
        date(2026, 12, 25),  # Christmas
        # Note: Diwali Laxmi Pujan (Nov 8, 2026) falls on a Sunday — Muhurat
        # trading only, not a weekday holiday, so it is omitted here.
    }
)

# Best-effort — confirm against the official NSE circular once published.
NSE_HOLIDAYS_2027: frozenset[date] = frozenset(
    {
        date(2027, 1, 26),  # Republic Day
        date(2027, 3, 22),  # Holi
        date(2027, 3, 26),  # Good Friday
        date(2027, 4, 14),  # Dr. Baba Saheb Ambedkar Jayanti
        date(2027, 4, 15),  # Shri Ram Navami
        date(2027, 10, 29),  # Diwali
        date(2027, 11, 15),  # Prakash Gurpurb Sri Guru Nanak Dev
        # Independence Day, Maharashtra Day, Gandhi Jayanti, Ganesh Chaturthi,
        # Dussehra and Christmas fall on weekends in 2027 and are already
        # excluded by the weekday check in Market.is_trading_day.
    }
)

NSE_HOLIDAYS: frozenset[date] = NSE_HOLIDAYS_2025 | NSE_HOLIDAYS_2026 | NSE_HOLIDAYS_2027
