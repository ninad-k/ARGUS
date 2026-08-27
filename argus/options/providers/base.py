"""Option-chain provider protocol and shared types.

Mirrors ``argus.data.prices.base``'s ``PriceDataProvider`` idiom (Protocol +
never-raise implementations), but keyed on ``Instrument`` rather than
``Market`` for ``supports`` -- option-chain availability is a per-symbol
question (not every symbol in an options-capable market actually has listed
options), unlike price data which is uniformly available across a market.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Protocol

from argus.data.prices.base import ProviderHealth
from argus.markets import Instrument
from argus.options.models import OptionChain


class OptionChainProvider(Protocol):
    """Async option-chain source. Implementations must never raise -- they
    log and return an empty/``None`` result on failure."""

    name: str

    def supports(self, inst: Instrument) -> bool:
        """Whether this provider can serve an option chain for ``inst``."""
        ...

    async def list_expiries(self, inst: Instrument) -> list[date]:
        """Return every expiry available for ``inst``. May be empty; never raises."""
        ...

    async def get_chain(self, inst: Instrument, expiry: date | None = None) -> OptionChain | None:
        """Return the option chain for ``inst`` at ``expiry``, or the nearest
        available expiry when ``expiry`` is ``None``. ``None`` if unavailable."""
        ...

    async def health_check(self) -> ProviderHealth:
        """Cheap self-check used by the source registry."""
        ...

    async def aclose(self) -> None:
        """Release any owned resources. Safe to call even if nothing needs closing."""
        ...


def nearest_expiry(expiries: list[date], *, today: date | None = None) -> date | None:
    """The soonest expiry on/after ``today`` (default: today's date), or the
    earliest expiry overall if every expiry has already passed. ``None`` if
    ``expiries`` is empty. Shared by every ``OptionChainProvider`` that needs
    to pick a default expiry for ``get_chain(inst, expiry=None)``."""
    if not expiries:
        return None
    ref = today if today is not None else date.today()  # noqa: DTZ011 -- expiry-day boundary only
    upcoming = [d for d in expiries if d >= ref]
    if upcoming:
        return min(upcoming)
    return min(expiries)


class StaticOptionChainProvider:
    """Serves chains from a fixed in-memory map, keyed by (symbol, market_code).
    Primarily for tests/fixtures."""

    name = "static"

    def __init__(self, chains: dict[tuple[str, str], OptionChain] | None = None) -> None:
        self._chains: dict[tuple[str, str], OptionChain] = chains or {}

    def add(self, chain: OptionChain) -> None:
        self._chains[(chain.symbol, chain.market_code)] = chain

    def supports(self, inst: Instrument) -> bool:
        return (inst.symbol, inst.market_code) in self._chains

    async def list_expiries(self, inst: Instrument) -> list[date]:
        chain = self._chains.get((inst.symbol, inst.market_code))
        return list(chain.expiries) if chain is not None else []

    async def get_chain(self, inst: Instrument, expiry: date | None = None) -> OptionChain | None:
        chain = self._chains.get((inst.symbol, inst.market_code))
        if chain is None:
            return None
        if expiry is None or expiry in chain.expiries:
            return chain
        return None

    async def health_check(self) -> ProviderHealth:
        return ProviderHealth(
            ok=True, detail="static provider always healthy", checked_at=datetime.now(UTC)
        )

    async def aclose(self) -> None:
        return None
