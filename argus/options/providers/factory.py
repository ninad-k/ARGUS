"""Factory for building an ``OptionChainProvider`` for an instrument's market."""

from __future__ import annotations

import structlog

from argus.markets import IN_NSE, US_NASDAQ, US_NYSE, Instrument
from argus.options.providers.base import OptionChainProvider
from argus.options.providers.nse_options import NSEOptionsProvider
from argus.options.providers.yfinance_options import YFinanceOptionsProvider

logger = structlog.get_logger(__name__)

_US_MARKET_CODES = frozenset({US_NYSE.code, US_NASDAQ.code})


def build_option_provider(inst: Instrument) -> OptionChainProvider | None:
    """Return the ``OptionChainProvider`` for ``inst``'s market, or ``None``
    for a market with no options-chain support wired up yet."""
    if inst.market_code == IN_NSE.code:
        return NSEOptionsProvider()
    if inst.market_code in _US_MARKET_CODES:
        return YFinanceOptionsProvider()
    logger.info("options.factory.unsupported_market", market=inst.market_code)
    return None
