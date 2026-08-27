"""Assembles OHLCV- (and, when available, options-chain-) derived orderflow
features for one instrument into a single ``OrderflowFeatures`` snapshot.

Everything here rides on the approximations documented in
``argus.orderflow.volume_profile``/``.liquidity``/``.gaps`` -- there is no
tick or L2 data behind any of it. The options-derived fields
(``pcr_oi``/``gex_sign``/``max_pain_dist_pct``) are ``None`` unless a
``chain`` is supplied -- fetching a chain is a network call, so callers only
pass one for instruments worth paying that cost for (see
``argus.pipeline``'s post-screen top-candidate annotation step).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

from argus.indicators.features import compute_features
from argus.options.analytics import gex_profile, max_pain, pcr
from argus.options.models import OptionChain
from argus.options.providers.base import nearest_expiry
from argus.orderflow.gaps import classify_gap
from argus.orderflow.liquidity import detect_sweeps
from argus.orderflow.volume_profile import volume_profile

# The JSON-serializable value type every OrderflowFeatures field (and hence
# to_feature_dict's output) is restricted to -- keeps it directly usable as
# a DailyPick.features_json["orderflow"] entry with no further conversion.
OrderflowFeatureValue = float | str | bool | None


@dataclass(frozen=True)
class OrderflowFeatures:
    """One instrument's orderflow snapshot -- see module docstring."""

    vp_poc: float | None
    vp_val: float | None
    vp_vah: float | None
    close_vs_poc_pct: float | None
    sweep: str
    sweep_reclaimed: bool | None
    gap_kind: str
    gap_pct: float | None
    gap_filled: bool | None
    rvol: float | None
    pcr_oi: float | None
    gex_sign: str | None
    max_pain_dist_pct: float | None


def compute_orderflow(
    daily_bars: NDArray[np.void],
    intraday_bars: NDArray[np.void] | None = None,
    chain: OptionChain | None = None,
) -> OrderflowFeatures | None:
    """Assemble volume-profile + sweep + gap + rvol (+ optional options-chain)
    features for the instrument ``daily_bars``/``intraday_bars`` belong to.

    ``None`` when ``daily_bars`` is empty -- there's nothing to compute
    anything from. Every other missing input (no intraday bars, no chain,
    too little history for a given sub-feature) degrades that sub-feature to
    ``None``/``"none"`` rather than making the whole result ``None``.

    The volume profile is built from ``intraday_bars`` when given (a finer
    price grid -- see ``argus.orderflow.volume_profile``), else falls back to
    ``daily_bars``.
    """
    if len(daily_bars) == 0:
        return None

    close = float(daily_bars[-1]["close"])

    vp_bars = (
        intraday_bars if intraday_bars is not None and len(intraday_bars) > 0 else daily_bars
    )
    vp = volume_profile(vp_bars)
    vp_poc = vp.poc if vp is not None else None
    vp_val = vp.value_area_low if vp is not None else None
    vp_vah = vp.value_area_high if vp is not None else None
    close_vs_poc_pct = ((close - vp_poc) / vp_poc * 100.0) if vp_poc else None

    sweeps = detect_sweeps(daily_bars)
    sweep = "none"
    sweep_reclaimed: bool | None = None
    if sweeps:
        # A bar can carry both a high and low sweep (an outside bar) -- a
        # low sweep is the long-biased signal this module's main consumer
        # (OrderflowConfluenceStrategy) cares about, so prefer it when both
        # fired; otherwise report whichever one did.
        chosen = next((s for s in sweeps if s.kind == "low_sweep"), sweeps[0])
        sweep = chosen.kind
        sweep_reclaimed = chosen.reclaimed

    gap = classify_gap(daily_bars)
    gap_kind = str(gap["kind"]) if gap is not None else "none"
    gap_pct = float(gap["gap_pct"]) if gap is not None else None
    gap_filled = bool(gap["filled"]) if gap is not None else None

    computed = compute_features(daily_bars)
    rvol_raw = computed.get("rvol", float("nan"))
    rvol = None if math.isnan(rvol_raw) else rvol_raw

    pcr_oi: float | None = None
    gex_sign: str | None = None
    max_pain_dist_pct: float | None = None
    if chain is not None and chain.expiries:
        expiry = nearest_expiry(chain.expiries)
        if expiry is not None:
            pcr_oi, _pcr_volume = pcr(chain, expiry)

            profile = gex_profile(chain, expiry, chain.spot)
            if profile:
                nearest = min(profile, key=lambda pair: abs(pair[0] - chain.spot))
                gex_sign = "long_gamma" if nearest[1] >= 0 else "short_gamma"

            pain = max_pain(chain, expiry)
            if pain:
                max_pain_dist_pct = (chain.spot - pain) / pain * 100.0

    return OrderflowFeatures(
        vp_poc=vp_poc,
        vp_val=vp_val,
        vp_vah=vp_vah,
        close_vs_poc_pct=close_vs_poc_pct,
        sweep=sweep,
        sweep_reclaimed=sweep_reclaimed,
        gap_kind=gap_kind,
        gap_pct=gap_pct,
        gap_filled=gap_filled,
        rvol=rvol,
        pcr_oi=pcr_oi,
        gex_sign=gex_sign,
        max_pain_dist_pct=max_pain_dist_pct,
    )


def to_feature_dict(of: OrderflowFeatures) -> dict[str, OrderflowFeatureValue]:
    """``OrderflowFeatures`` as a plain JSON-serializable dict, for storing
    under ``Candidate.features["orderflow"]`` (-> ``DailyPick.features_json``)."""
    return asdict(of)


def format_orderflow_summary(of: dict[str, OrderflowFeatureValue]) -> str:
    """Render ``to_feature_dict``'s output as one compact human-readable
    line, e.g. ``"POC 182.40, +2.1% above; low sweep reclaimed; gap up 1.2%
    unfilled; PCR 0.84; long gamma"``. Shared by the markdown report, the LLM
    digest, and the picks UI so the three renderings can't drift apart.
    Empty string when ``of`` carries nothing renderable.
    """
    parts: list[str] = []

    poc = of.get("vp_poc")
    close_vs_poc = of.get("close_vs_poc_pct")
    if isinstance(poc, float) and isinstance(close_vs_poc, float):
        direction = "above" if close_vs_poc >= 0 else "below"
        parts.append(f"POC {poc:.2f}, {close_vs_poc:+.1f}% {direction}")

    sweep = of.get("sweep")
    if isinstance(sweep, str) and sweep != "none":
        label = sweep.replace("_", " ")
        reclaimed = of.get("sweep_reclaimed")
        if reclaimed is True:
            parts.append(f"{label} reclaimed")
        elif reclaimed is False:
            parts.append(f"{label} held")
        else:
            parts.append(label)

    gap_kind = of.get("gap_kind")
    if isinstance(gap_kind, str) and gap_kind != "none":
        bits = [gap_kind.replace("_", " ")]
        gap_pct = of.get("gap_pct")
        if isinstance(gap_pct, float):
            bits.append(f"{gap_pct:.1f}%")
        filled = of.get("gap_filled")
        if filled is True:
            bits.append("filled")
        elif filled is False:
            bits.append("unfilled")
        parts.append(" ".join(bits))

    pcr_oi = of.get("pcr_oi")
    if isinstance(pcr_oi, float):
        parts.append(f"PCR {pcr_oi:.2f}")

    gex_sign = of.get("gex_sign")
    if isinstance(gex_sign, str):
        parts.append(gex_sign.replace("_", " "))

    return "; ".join(parts)
