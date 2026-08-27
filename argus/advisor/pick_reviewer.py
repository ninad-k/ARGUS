"""Single-pass LLM review of a screen result.

One batch call reviews every candidate from a ``run_screen()`` result at once
(local models are slow — a per-pick call would make a daily run impractical).
The LLM's response is parsed defensively: any failure — backend down, bad
JSON, unexpected fields — degrades to an empty verdict map rather than
breaking the screener pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import structlog

from argus.advisor.llm import LLMBackend, LLMRequest, parse_llm_json
from argus.markets import Market
from argus.screener.base import Candidate
from argus.screener.runner import ScreenResult

logger = structlog.get_logger(__name__)

Verdict = Literal["buy", "watch", "avoid"]
_VALID_VERDICTS = frozenset({"buy", "watch", "avoid"})

# Compact feature subset shown to the LLM per candidate — enough to reason
# about momentum/extension without dumping the full feature vector.
_PROMPT_FEATURE_KEYS: tuple[str, ...] = (
    "close",
    "rsi_14",
    "roc_20",
    "roc_60",
    "rvol",
    "pct_off_high",
    "dist_sma200_pct",
)

_SYSTEM_PROMPT = (
    "You are a senior equity analyst reviewing candidates produced by a "
    "quantitative stock screener. Be concise and skeptical of overextended "
    "or low-conviction setups. Respond with strict JSON only, no markdown "
    'fencing and no prose outside the JSON. The JSON must be a single object '
    'of the form {"picks": [{"symbol": str, "verdict": "buy"|"watch"|"avoid", '
    '"confidence": int (0-100), "thesis": str, "risks": str}, ...]}, with '
    "exactly one entry per candidate listed below, using the given symbols."
)


@dataclass(frozen=True)
class PickVerdict:
    symbol: str
    verdict: Verdict
    confidence: int
    thesis: str
    risks: str


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
    for key in _PROMPT_FEATURE_KEYS:
        value = c.features.get(key)
        if value is None or math.isnan(value):
            continue
        feature_bits.append(f"{key}={value:.2f}")
    if feature_bits:
        lines.append("Features: " + ", ".join(feature_bits))

    return "\n".join(lines)


def _build_user_prompt(candidates: list[Candidate], market: Market) -> str:
    header = f"Market: {market.name} ({market.code})\nCandidates ({len(candidates)}):\n\n"
    blocks = "\n\n".join(_format_candidate(c) for c in candidates)
    return header + blocks


def _coerce_verdict(raw: Any) -> Verdict:
    if raw in _VALID_VERDICTS:
        verdict: Verdict = raw
        return verdict
    return "watch"


def _coerce_confidence(raw: Any) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return 0
    return max(0, min(100, value))


def _parse_rows(text: str) -> list[Any]:
    """Extract the ``picks`` array from the LLM's JSON object response."""
    parsed = parse_llm_json(text)
    if parsed is None:
        return []
    rows = parsed.get("picks")
    if not isinstance(rows, list):
        return []
    return rows


async def review_picks(
    candidates: list[Candidate], market: Market, backend: LLMBackend
) -> dict[str, PickVerdict]:
    """Review ``candidates`` in a single LLM call and return per-symbol verdicts.

    Never raises: a backend failure, timeout, or unparseable/empty response
    (including a NoOp backend's empty string) all result in ``{}`` with a
    logged warning — the caller's screener pipeline must keep running.
    """
    if not candidates:
        return {}

    request = LLMRequest(system=_SYSTEM_PROMPT, user=_build_user_prompt(candidates, market))
    try:
        response = await backend.complete(request)
    except Exception as exc:  # LLM failure must never break the screener pipeline
        logger.warning("advisor.pick_reviewer.backend_failed", error=str(exc))
        return {}

    if not response.text:
        logger.warning("advisor.pick_reviewer.empty_response", provider=response.provider)
        return {}

    rows = _parse_rows(response.text)
    if not rows:
        logger.warning("advisor.pick_reviewer.unparseable_response", provider=response.provider)
        return {}

    valid_symbols = {c.instrument.symbol for c in candidates}
    verdicts: dict[str, PickVerdict] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        symbol = row.get("symbol")
        if not isinstance(symbol, str) or symbol not in valid_symbols:
            continue
        verdicts[symbol] = PickVerdict(
            symbol=symbol,
            verdict=_coerce_verdict(row.get("verdict")),
            confidence=_coerce_confidence(row.get("confidence")),
            thesis=str(row.get("thesis") or ""),
            risks=str(row.get("risks") or ""),
        )
    return verdicts


def apply_verdicts(result: ScreenResult, verdicts: dict[str, PickVerdict]) -> None:
    """Annotate ``result``'s candidates in place with their LLM verdict, if any.

    ``result.top`` holds references into ``result.candidates`` (it's a
    slice), so mutating the candidates here is visible through both.
    """
    for c in result.candidates:
        verdict = verdicts.get(c.instrument.symbol)
        if verdict is not None:
            c.llm_verdict = verdict
