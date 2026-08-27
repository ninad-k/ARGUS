"""Open-interest, gamma-exposure, PCR, max-pain, and IV-rank analytics.

``oi_profile``/``pcr``/``gex_profile`` are ported from DRUVA's
``core/options/oi_analytics.py`` (``oi_profile``/``summarise``), reshaped to
read ARGUS's flat ``OptionChain.quotes`` (one row per strike/expiry/right)
instead of DRUVA's ``OptionChain.rows`` (one row per strike, CE/PE nested)
-- the underlying math (GEX's dealer-hedge sign convention, PCR as
put/call ratio) is unchanged. ``iv_rank`` is ported from
``core/options/iv_rank.py`` (``IVRankCalculator.compute_iv_rank``), likewise
math-unchanged.

``max_pain`` has no DRUVA source (``oi_analytics.py``/``iv_rank.py`` don't
implement it) -- it's a fresh implementation of the standard "max pain"
definition: the strike at which option writers, in aggregate, owe the least
intrinsic value at expiry.

GEX sign convention (kept exactly as DRUVA's ``oi_profile`` computed it):
dealers are modeled as short calls / short puts (the standard dealer-hedge
assumption), so a strike's dealer gamma exposure is
``(-call_oi * call_gamma + put_oi * put_gamma) * spot^2 * 0.01`` -- calls
contribute *negative* GEX, puts *positive*, at a given strike (both
``call_gamma``/``put_gamma`` are non-negative, so the sign comes entirely
from OI side and the leading minus/plus).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from argus.options.models import OptionChain


@dataclass(frozen=True, slots=True)
class OiProfileEntry:
    """Per-strike call/put open interest and volume for one expiry."""

    strike: float
    call_oi: float
    put_oi: float
    call_volume: float
    put_volume: float


def oi_profile(chain: OptionChain, expiry: date) -> list[OiProfileEntry]:
    """Per-strike call/put OI + volume for ``expiry``, sorted by strike.
    Missing OI/volume (``None``) is treated as 0."""
    calls_by_strike = {q.strike: q for q in chain.calls(expiry)}
    puts_by_strike = {q.strike: q for q in chain.puts(expiry)}
    strikes = sorted(set(calls_by_strike) | set(puts_by_strike))

    out: list[OiProfileEntry] = []
    for strike in strikes:
        c = calls_by_strike.get(strike)
        p = puts_by_strike.get(strike)
        out.append(
            OiProfileEntry(
                strike=strike,
                call_oi=(c.oi if c and c.oi is not None else 0.0),
                put_oi=(p.oi if p and p.oi is not None else 0.0),
                call_volume=(c.volume if c and c.volume is not None else 0.0),
                put_volume=(p.volume if p and p.volume is not None else 0.0),
            )
        )
    return out


def pcr(chain: OptionChain, expiry: date) -> tuple[float, float]:
    """Put-call ratio for ``expiry``: ``(pcr_by_oi, pcr_by_volume)``.

    Each ratio is put/call; 0.0 when the call side is empty (avoids a
    division by zero on an all-put or empty chain), matching DRUVA's
    ``summarise``."""
    profile = oi_profile(chain, expiry)
    total_call_oi = sum(p.call_oi for p in profile)
    total_put_oi = sum(p.put_oi for p in profile)
    total_call_vol = sum(p.call_volume for p in profile)
    total_put_vol = sum(p.put_volume for p in profile)
    pcr_oi = (total_put_oi / total_call_oi) if total_call_oi else 0.0
    pcr_volume = (total_put_vol / total_call_vol) if total_call_vol else 0.0
    return pcr_oi, pcr_volume


def gex_profile(chain: OptionChain, expiry: date, spot: float) -> list[tuple[float, float]]:
    """Per-strike dealer gamma exposure for ``expiry`` at a given ``spot``
    (pass ``chain.spot`` for the chain's own spot, or a hypothetical spot to
    see how GEX would look elsewhere). Returns ``(strike, gex)`` pairs sorted
    by strike. Missing gamma/OI (``None``) is treated as 0 -- see module
    docstring for the sign convention."""
    calls_by_strike = {q.strike: q for q in chain.calls(expiry)}
    puts_by_strike = {q.strike: q for q in chain.puts(expiry)}
    strikes = sorted(set(calls_by_strike) | set(puts_by_strike))

    out: list[tuple[float, float]] = []
    for strike in strikes:
        c = calls_by_strike.get(strike)
        p = puts_by_strike.get(strike)
        call_oi = c.oi if c and c.oi is not None else 0.0
        put_oi = p.oi if p and p.oi is not None else 0.0
        call_gamma = c.gamma if c and c.gamma is not None else 0.0
        put_gamma = p.gamma if p and p.gamma is not None else 0.0
        gex = (-call_oi * call_gamma + put_oi * put_gamma) * spot * spot * 0.01
        out.append((strike, gex))
    return out


def max_pain(chain: OptionChain, expiry: date) -> float:
    """The strike at which option writers, in aggregate, owe the least total
    intrinsic value at expiry -- the standard "max pain" definition (not
    ported from DRUVA; see module docstring).

    Candidates are the chain's own traded strikes for ``expiry``. Falls back
    to ``chain.spot`` when there are no strikes to evaluate."""
    profile = oi_profile(chain, expiry)
    if not profile:
        return chain.spot

    def payout_at(candidate: float) -> float:
        total = 0.0
        for entry in profile:
            if entry.strike <= candidate:
                total += (candidate - entry.strike) * entry.call_oi
            if entry.strike >= candidate:
                total += (entry.strike - candidate) * entry.put_oi
        return total

    return min((entry.strike for entry in profile), key=payout_at)


def iv_rank(current_iv: float, iv_history: Sequence[float]) -> float:
    """IV Rank: ``(current - low) / (high - low) * 100``, clamped to
    ``[0, 100]``. Ported from DRUVA's ``IVRankCalculator.compute_iv_rank``.
    Returns 0.0 if ``iv_history`` is empty or its range is ~zero."""
    if not iv_history:
        return 0.0
    low = min(iv_history)
    high = max(iv_history)
    if high - low < 1e-9:
        return 0.0
    rank = (current_iv - low) / (high - low) * 100.0
    return max(0.0, min(100.0, rank))


def atm_iv(chain: OptionChain, expiry: date) -> float | None:
    """Average of the call/put IV at the ATM strike for ``expiry`` (mirrors
    DRUVA's ``fetch_nifty_iv``, which averages CE+PE IV at the ATM strike).
    ``None`` if there's no ATM strike or neither side has an IV."""
    atm = chain.atm_strike(expiry)
    if atm is None:
        return None
    ivs = [q.iv for q in chain.for_expiry(expiry) if q.strike == atm and q.iv is not None]
    if not ivs:
        return None
    return sum(ivs) / len(ivs)
