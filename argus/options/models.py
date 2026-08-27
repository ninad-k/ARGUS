"""Market-agnostic option chain data model.

Adapted from DRUVA's ``core/options/option_chain.py`` (``OptionLeg`` /
``OptionChainRow`` -- a CE/PE pair keyed by strike, one chain per expiry).
Reshaped here into a flat list of per-(strike, expiry, right) quotes so a
single ``OptionChain`` can carry whatever expiries a provider's fetch
actually populated, and so US (``C``/``P``) and Indian (``CE``/``PE``)
conventions collapse onto one ``Right`` literal instead of two DTOs. The
``for_expiry``/``calls``/``puts``/``atm_strike`` helpers below give back the
same "one row per strike, CE + PE side-by-side" view DRUVA's
``OptionChainRow`` provided, computed on demand instead of stored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Literal

Right = Literal["C", "P"]


@dataclass(frozen=True, slots=True)
class OptionQuote:
    """A single option contract's market data + (optionally) computed greeks.

    ``iv``/``delta``/``gamma``/``theta``/``vega`` are ``None`` when the
    provider had nothing to compute them from (e.g. no trade, no quote) --
    callers should treat a ``None`` greek as "unavailable", not zero.
    """

    strike: float
    expiry: date
    right: Right
    bid: float | None = None
    ask: float | None = None
    last: float | None = None
    iv: float | None = None
    oi: float | None = None
    volume: float | None = None
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None


@dataclass(slots=True)
class OptionChain:
    """A snapshot of option quotes for one underlying symbol.

    ``expiries`` lists every expiry the provider knows about for this
    underlying (informational -- e.g. for a UI expiry picker); ``quotes`` may
    cover only a subset of those expiries when a provider fetches one expiry
    at a time (yfinance) rather than all of them in one response (NSE) -- use
    ``for_expiry``/``calls``/``puts`` rather than assuming ``quotes`` spans
    every listed expiry.
    """

    symbol: str
    market_code: str
    spot: float
    as_of: datetime
    expiries: list[date] = field(default_factory=list)
    quotes: list[OptionQuote] = field(default_factory=list)

    def for_expiry(self, expiry: date) -> list[OptionQuote]:
        """All quotes (calls and puts) for ``expiry``."""
        return [q for q in self.quotes if q.expiry == expiry]

    def calls(self, expiry: date) -> list[OptionQuote]:
        return [q for q in self.for_expiry(expiry) if q.right == "C"]

    def puts(self, expiry: date) -> list[OptionQuote]:
        return [q for q in self.for_expiry(expiry) if q.right == "P"]

    def strikes(self, expiry: date) -> list[float]:
        """Sorted, deduplicated strikes quoted for ``expiry``."""
        return sorted({q.strike for q in self.for_expiry(expiry)})

    def atm_strike(self, expiry: date) -> float | None:
        """The strike closest to ``spot`` for ``expiry``, or ``None`` if
        there are no quotes for that expiry."""
        strikes = self.strikes(expiry)
        if not strikes:
            return None
        return min(strikes, key=lambda k: abs(k - self.spot))

    @staticmethod
    def mid(quote: OptionQuote) -> float | None:
        """Bid/ask midpoint, falling back to ``last`` when either side of
        the quote is missing (thin/illiquid strikes often have no live
        bid/ask), and to ``None`` when nothing is available at all."""
        if quote.bid is not None and quote.ask is not None:
            return (quote.bid + quote.ask) / 2.0
        return quote.last
