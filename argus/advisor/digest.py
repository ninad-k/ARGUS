"""Shared candidate-digest builder for LLM review calls.

Both the single-pass reviewer (``pick_reviewer.review_picks``) and the
persona council (``council.council_review``) send the *same* candidate
digest to the LLM -- only the system prompt differs (a neutral analyst vs. a
named persona's voice). Factored out here so the two callers can't drift.
"""

from __future__ import annotations

import math

from argus.markets import Market
from argus.orderflow.features import format_orderflow_summary
from argus.screener.base import Candidate

# Compact feature subset shown to the LLM per candidate -- enough to reason
# about momentum/extension without dumping the full feature vector.
PROMPT_FEATURE_KEYS: tuple[str, ...] = (
    "close",
    "rsi_14",
    "roc_20",
    "roc_60",
    "rvol",
    "pct_off_high",
    "dist_sma200_pct",
)


def _format_candidate(c: Candidate) -> str:
    lines = [
        f"Symbol: {c.instrument.symbol}",
        f"Strategy: {c.strategy}",
        f"Score: {c.score:.2f}",
        f"Direction: {c.direction}",
        f"Stage: {c.stage}",
        f"Reason: {c.reason}",
    ]
    if c.entry is not None:
        lines.append(f"Entry: {c.entry:.2f}")
    if c.stop is not None:
        lines.append(f"Stop: {c.stop:.2f}")
    if c.target is not None:
        lines.append(f"Target: {c.target:.2f}")

    feature_bits = []
    for key in PROMPT_FEATURE_KEYS:
        value = c.features.get(key)
        if value is None or math.isnan(value):
            continue
        feature_bits.append(f"{key}={value:.2f}")
    if feature_bits:
        lines.append("Features: " + ", ".join(feature_bits))

    orderflow = c.features.get("orderflow")
    if isinstance(orderflow, dict):
        orderflow_line = format_orderflow_summary(orderflow)
        if orderflow_line:
            lines.append("Orderflow: " + orderflow_line)

    return "\n".join(lines)


def build_digest(candidates: list[Candidate], market: Market) -> str:
    """Render ``candidates`` into the shared user-prompt digest format."""
    header = f"Market: {market.name} ({market.code})\nCandidates ({len(candidates)}):\n\n"
    blocks = "\n\n".join(_format_candidate(c) for c in candidates)
    return header + blocks
