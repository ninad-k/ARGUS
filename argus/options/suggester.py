"""Risk-based derivative (option strike / futures) suggester.

Given one screener ``Candidate`` and an option chain for its instrument,
``suggest_derivative`` picks a single strike (CE for a long call, PE for a
short/put idea) or, when the underlying has listed futures and implied vol
looks expensive, a futures idea instead -- all analysis-only, no order of
any kind is ever placed from this module or anything downstream of it (see
``argus.paper`` -- the paper engine only ever trades equities).

``suggest_for_picks`` fans this out across a ``ScreenReport``'s top picks,
building a fresh ``OptionChainProvider`` per instrument via
``argus.options.providers.factory.build_option_provider`` (or an injected
``provider_factory`` for tests), and ``persist_suggestions`` writes the
result to the ``option_suggestions`` table, linked to the matching
``DailyPick`` row for the run.

Every public async function here is written to never raise -- a suggestion
failure (bad chain, provider error, missing greeks) degrades to "no
suggestion for this symbol", never to a pipeline failure.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time
from enum import StrEnum
from typing import TYPE_CHECKING, Literal

import structlog
from sqlalchemy import select

from argus.config import AppSettings, OptionsSettings, get_settings
from argus.db import async_session
from argus.db.models import DailyPick, OptionSuggestion
from argus.markets import Instrument
from argus.options.analytics import atm_iv, iv_rank
from argus.options.models import OptionChain, OptionQuote, Right
from argus.options.providers.base import OptionChainProvider
from argus.options.providers.factory import build_option_provider
from argus.screener.base import Candidate

if TYPE_CHECKING:
    from argus.pipeline import ScreenReport

logger = structlog.get_logger(__name__)

# Bounds how many concurrent (provider-build + chain-fetch) calls
# ``suggest_for_picks`` issues at once -- mirrors
# ``argus.data.fundamentals.base._DEFAULT_GET_MANY_CONCURRENCY``'s reasoning:
# a run's top picks could span many symbols, and providers hit a live
# network endpoint per instrument.
_CONCURRENCY = 3

# Documented approximation: a long futures position needs margin, not the
# full notional, so ``est_cost`` for a FUT suggestion is a rough proxy --
# not a real broker margin calculation -- for "how much capital this idea
# roughly ties up".
_FUTURES_MARGIN_PROXY_PCT = 0.15

_NA = "NA"


class RiskLevel(StrEnum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


# (low, high) absolute-delta band per risk level. Conservative picks deep
# in-the-money strikes (high delta, behaves closer to the underlying, less
# time-value risk); aggressive picks far out-of-the-money strikes (low
# delta, cheap premium, higher leverage/risk of expiring worthless).
_DELTA_BANDS: dict[RiskLevel, tuple[float, float]] = {
    RiskLevel.CONSERVATIVE: (0.55, 0.70),
    RiskLevel.MODERATE: (0.35, 0.50),
    RiskLevel.AGGRESSIVE: (0.15, 0.30),
}

_RIGHT_TO_INSTRUMENT_TYPE: dict[Right, Literal["CE", "PE"]] = {"C": "CE", "P": "PE"}


@dataclass(frozen=True, slots=True)
class DerivativeSuggestion:
    """One derivative idea attached to a screener pick.

    ``strike``/``iv``/``delta``/``oi`` are ``None`` for a futures
    suggestion (``instrument_type="FUT"``) -- futures have no strike or
    option greeks. ``est_cost`` is ``suggested_price * lot_size`` for an
    option (the actual premium outlay) but a margin *proxy* for futures --
    see ``_FUTURES_MARGIN_PROXY_PCT``.
    """

    symbol: str
    market_code: str
    instrument_type: Literal["CE", "PE", "FUT"]
    strike: float | None
    expiry: date
    suggested_price: float | None
    iv: float | None
    delta: float | None
    oi: float | None
    lot_size: int
    est_cost: float | None
    rationale: str
    risk_level: RiskLevel


def select_expiry(chain: OptionChain, min_days: int = 20, max_days: int = 60) -> date | None:
    """Pick the expiry to suggest against.

    Preference order: the nearest expiry with ``min_days <= DTE <= max_days``;
    failing that, the nearest expiry with ``DTE >= min_days`` (further out
    than the preferred window, but at least not expiring too soon); failing
    that, the nearest expiry available at all (everything quoted expires
    sooner than ``min_days``). ``None`` if the chain lists no expiries.
    """
    if not chain.expiries:
        return None
    today = date.today()  # noqa: DTZ011 -- DTE is a calendar-day count, not a market-tz boundary

    def dte(expiry: date) -> int:
        return (expiry - today).days

    in_window = [e for e in chain.expiries if min_days <= dte(e) <= max_days]
    if in_window:
        return min(in_window, key=dte)

    at_least_min = [e for e in chain.expiries if dte(e) >= min_days]
    if at_least_min:
        return min(at_least_min, key=dte)

    # Every expiry is shorter-dated than min_days -- "nearest available"
    # means closest to the desired window, i.e. the longest-dated of the
    # too-soon expiries (least short of min_days), not necessarily the
    # soonest one.
    return min(chain.expiries, key=lambda e: abs(dte(e) - min_days))


def _passes_liquidity(quote: OptionQuote, *, min_oi: int, max_spread_pct: float) -> bool:
    if quote.oi is None or quote.oi < min_oi:
        return False
    if quote.bid is None or quote.ask is None:
        return False
    mid = (quote.bid + quote.ask) / 2.0
    if mid <= 0:
        return False
    spread_pct = (quote.ask - quote.bid) / mid * 100.0
    return spread_pct <= max_spread_pct


def _closest_to_midpoint(quotes: Sequence[OptionQuote], band_mid: float) -> OptionQuote:
    return min(quotes, key=lambda q: abs(abs(q.delta or 0.0) - band_mid))


def _deepest_itm(quotes: Sequence[OptionQuote]) -> OptionQuote:
    return max(quotes, key=lambda q: abs(q.delta or 0.0))


def _depth_label(quote: OptionQuote, spot: float, right: Right) -> str:
    if right == "C":
        if quote.strike < spot * 0.99:
            return "ITM"
        if quote.strike > spot * 1.01:
            return "OTM"
        return "ATM"
    if quote.strike > spot * 1.01:
        return "ITM"
    if quote.strike < spot * 0.99:
        return "OTM"
    return "ATM"


def _oi_label(oi: float | None) -> str:
    if oi is None:
        return _NA
    if oi >= 1000:
        return f"{oi / 1000.0:.1f}k"
    return f"{oi:.0f}"


def _ivr_label(ivr: float | None) -> str:
    return f"{ivr:.0f}" if ivr is not None else _NA


def _rationale(
    *,
    risk: RiskLevel,
    symbol: str,
    expiry: date,
    strike: float,
    right: Right,
    price: float | None,
    delta: float | None,
    oi: float | None,
    ivr: float | None,
    depth: str,
    dte: int,
    notes: Sequence[str] = (),
) -> str:
    right_label = "C" if right == "C" else "P"
    price_label = f"~{price:.2f}" if price is not None else _NA
    delta_label = f"{delta:.2f}" if delta is not None else _NA
    kind = "call" if right == "C" else "put"
    base = (
        f"{risk.value.upper()}: {symbol} {expiry:%d %b} {strike:g}{right_label} "
        f"@ {price_label}, Δ{delta_label}, OI {_oi_label(oi)}, IVR {_ivr_label(ivr)} "
        f"— {depth} {kind}, {dte} DTE"
    )
    if notes:
        base += " (" + "; ".join(notes) + ")"
    return base


def suggest_derivative(
    candidate: Candidate,
    chain: OptionChain,
    *,
    risk: RiskLevel,
    settings: OptionsSettings,
    iv_history: Sequence[float] = (),
) -> DerivativeSuggestion | None:
    """Pick one derivative idea for ``candidate`` from ``chain``, or ``None``
    when nothing suitable is available. See module docstring for the
    selection logic. Never raises."""
    inst = candidate.instrument
    if not inst.has_options and not inst.has_futures:
        logger.debug("options.suggester.no_derivatives", symbol=inst.symbol)
        return None

    expiry = select_expiry(chain, settings.expiry_min_days, settings.expiry_max_days)
    if expiry is None:
        logger.debug("options.suggester.no_expiry", symbol=inst.symbol)
        return None

    today = date.today()  # noqa: DTZ011 -- DTE is a calendar-day count
    dte = max((expiry - today).days, 0)

    atm_iv_value = atm_iv(chain, expiry)
    # IV-Rank guard is skipped entirely when the caller has no history to
    # rank against (documented in the task spec) -- ``ivr`` stays ``None``
    # rather than defaulting to 0, so "expensive" checks below never fire.
    ivr = iv_rank(atm_iv_value, iv_history) if iv_history and atm_iv_value is not None else None
    ivr_expensive = ivr is not None and ivr > settings.ivr_expensive_threshold

    if inst.has_futures and ivr_expensive and risk is RiskLevel.CONSERVATIVE:
        return _suggest_futures(inst, chain, expiry, risk, ivr, dte)

    if not inst.has_options:
        logger.debug("options.suggester.no_options", symbol=inst.symbol)
        return None

    right: Right = "C" if candidate.direction == "long" else "P"
    band_lo, band_hi = _DELTA_BANDS[risk]
    band_mid = (band_lo + band_hi) / 2.0

    in_band = [
        q
        for q in chain.for_expiry(expiry)
        if q.right == right and q.delta is not None and band_lo <= abs(q.delta) <= band_hi
    ]
    if not in_band:
        logger.debug("options.suggester.no_deltas_in_band", symbol=inst.symbol, right=right)
        return None

    liquid = [
        q
        for q in in_band
        if _passes_liquidity(q, min_oi=settings.min_oi, max_spread_pct=settings.max_spread_pct)
    ]
    notes: list[str] = []
    if not liquid:
        liquid = [q for q in in_band if q.last is not None]
        if not liquid:
            logger.debug("options.suggester.no_liquid_quotes", symbol=inst.symbol)
            return None
        notes.append("thin liquidity -- relaxed filter")

    if ivr_expensive and risk is not RiskLevel.AGGRESSIVE:
        # Conservative/moderate: expensive premium -> prefer the deepest ITM
        # (highest |delta|) quote still in-band rather than the usual
        # midpoint pick, since a cheaper OTM strike pays a disproportionate
        # amount of that rich implied vol for its extrinsic value.
        quote = _deepest_itm(liquid)
        notes.append(f"elevated IV (IVR {ivr:.0f}) -- deeper ITM preferred")
    else:
        quote = _closest_to_midpoint(liquid, band_mid)
        if ivr_expensive:
            # Aggressive: still allowed, just flagged -- the whole point of
            # an aggressive idea is cheap leveraged premium, so this risk
            # level doesn't trade that away for a deeper (pricier) strike.
            notes.append(f"elevated IV (IVR {ivr:.0f})")

    suggested_price = OptionChain.mid(quote)
    if suggested_price is None:
        suggested_price = quote.last

    instrument_type = _RIGHT_TO_INSTRUMENT_TYPE[right]
    est_cost = suggested_price * inst.lot_size if suggested_price is not None else None
    depth = _depth_label(quote, chain.spot, right)
    rationale = _rationale(
        risk=risk,
        symbol=inst.symbol,
        expiry=expiry,
        strike=quote.strike,
        right=right,
        price=suggested_price,
        delta=quote.delta,
        oi=quote.oi,
        ivr=ivr,
        depth=depth,
        dte=dte,
        notes=notes,
    )

    return DerivativeSuggestion(
        symbol=inst.symbol,
        market_code=inst.market_code,
        instrument_type=instrument_type,
        strike=quote.strike,
        expiry=expiry,
        suggested_price=suggested_price,
        iv=quote.iv,
        delta=quote.delta,
        oi=quote.oi,
        lot_size=inst.lot_size,
        est_cost=est_cost,
        rationale=rationale,
        risk_level=risk,
    )


def _suggest_futures(
    inst: Instrument,
    chain: OptionChain,
    expiry: date,
    risk: RiskLevel,
    ivr: float | None,
    dte: int,
) -> DerivativeSuggestion:
    ivr_label = f"{ivr:.0f}" if ivr is not None else _NA
    rationale = (
        f"{risk.value.upper()}: {inst.symbol} {expiry:%d %b} futures @ ~{chain.spot:.2f} "
        f"— IVR {ivr_label} is elevated, avoiding long option premium; futures give "
        f"directional exposure without paying rich implied vol, {dte} DTE"
    )
    est_cost = chain.spot * inst.lot_size * _FUTURES_MARGIN_PROXY_PCT
    return DerivativeSuggestion(
        symbol=inst.symbol,
        market_code=inst.market_code,
        instrument_type="FUT",
        strike=None,
        expiry=expiry,
        suggested_price=chain.spot,
        iv=None,
        delta=None,
        oi=None,
        lot_size=inst.lot_size,
        est_cost=est_cost,
        rationale=rationale,
        risk_level=risk,
    )


ProviderFactory = Callable[[Instrument], OptionChainProvider | None]
IvHistoryLookup = Callable[[Instrument], Sequence[float]]


async def suggest_for_picks(
    report: ScreenReport,
    *,
    risk: RiskLevel,
    settings: OptionsSettings | None = None,
    iv_history_lookup: IvHistoryLookup | None = None,
    provider_factory: ProviderFactory | None = None,
) -> dict[str, DerivativeSuggestion]:
    """Build a ``DerivativeSuggestion`` for each of ``report.result.top`` whose
    instrument has options or futures, keyed by symbol.

    A pick whose instrument has neither is skipped without ever touching a
    provider -- this is what keeps offline pipeline tests (synthetic
    instruments, ``has_options=False``/``has_futures=False``) network-free.
    ``provider_factory`` defaults to
    ``argus.options.providers.factory.build_option_provider`` and is the
    injection point tests use to serve a ``StaticOptionChainProvider``
    instead. Bounded concurrency via a semaphore (see ``_CONCURRENCY``).
    Never raises -- a per-symbol provider/build/fetch failure is logged and
    that symbol is simply absent from the result.
    """
    opts = settings if settings is not None else get_settings().options
    factory = provider_factory if provider_factory is not None else build_option_provider
    semaphore = asyncio.Semaphore(_CONCURRENCY)
    results: dict[str, DerivativeSuggestion] = {}

    async def _one(candidate: Candidate) -> None:
        inst = candidate.instrument
        if not inst.has_options and not inst.has_futures:
            return
        # ``provider`` is built and cleaned up inside the try/finally too --
        # a misbehaving factory (or a provider constructor that raises) must
        # not escape this function any more than a bad fetch would.
        provider: OptionChainProvider | None = None
        try:
            provider = factory(inst)
            if provider is None:
                return
            async with semaphore:
                chain = await provider.get_chain(inst)
                if chain is None:
                    return
                history = iv_history_lookup(inst) if iv_history_lookup is not None else ()
                suggestion = suggest_derivative(
                    candidate, chain, risk=risk, settings=opts, iv_history=history
                )
            if suggestion is not None:
                results[inst.symbol] = suggestion
        except Exception as exc:  # noqa: BLE001 -- this path must never raise
            logger.warning(
                "options.suggester.suggest_for_picks.failed",
                symbol=inst.symbol,
                error=str(exc),
            )
        finally:
            if provider is not None:
                try:
                    await provider.aclose()
                except Exception as exc:  # noqa: BLE001 -- this path must never raise
                    logger.warning(
                        "options.suggester.suggest_for_picks.aclose_failed",
                        symbol=inst.symbol,
                        error=str(exc),
                    )

    await asyncio.gather(*(_one(c) for c in report.result.top))
    return results


async def persist_suggestions(
    run_id: int,
    suggestions: dict[str, DerivativeSuggestion],
    risk: RiskLevel,
    settings: AppSettings | None = None,
) -> int:
    """Insert one ``OptionSuggestion`` row per entry in ``suggestions``,
    linked to the matching ``DailyPick`` for ``run_id`` by symbol. A symbol
    with no matching pick in this run is skipped. Returns the number of rows
    inserted. Never raises.

    ``OptionSuggestion.strike`` is a non-nullable column (committed schema,
    not revisited by this task); a futures suggestion (``strike=None`` on
    ``DerivativeSuggestion``) is stored as ``0.0`` -- a documented sentinel,
    distinguishable from a real strike by ``instrument_type == "FUT"``.
    """
    if not suggestions:
        return 0
    try:
        async with async_session(settings) as session:
            picks_result = await session.execute(
                select(DailyPick).where(DailyPick.run_id == run_id)
            )
            picks_by_symbol = {p.symbol: p for p in picks_result.scalars().all()}

            inserted = 0
            for symbol, suggestion in suggestions.items():
                pick = picks_by_symbol.get(symbol)
                if pick is None:
                    logger.debug(
                        "options.suggester.persist.no_matching_pick", symbol=symbol, run_id=run_id
                    )
                    continue
                session.add(
                    OptionSuggestion(
                        pick_id=pick.id,
                        risk_level=risk.value,
                        instrument_type=suggestion.instrument_type,
                        strike=suggestion.strike if suggestion.strike is not None else 0.0,
                        expiry=datetime.combine(suggestion.expiry, time.min),
                        suggested_price=suggestion.suggested_price,
                        iv=suggestion.iv,
                        delta=suggestion.delta,
                        oi=int(suggestion.oi) if suggestion.oi is not None else None,
                        rationale=suggestion.rationale,
                    )
                )
                inserted += 1
            await session.commit()
            return inserted
    except Exception as exc:  # noqa: BLE001 -- this path must never raise
        logger.warning("options.suggester.persist_suggestions.failed", error=str(exc))
        return 0
