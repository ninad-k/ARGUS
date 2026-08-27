"""Investor-persona council — fan a batch of candidates out to N personas in
parallel, then fuse their votes deterministically (no second LLM round-trip).

Mirrors DRUVA's ``Council`` idiom (see
``DRUVA/backend/app/core/advisor/council.py``): each persona reasons once,
independently, in its own voice; a plain confidence-weighted vote across
personas -- not another LLM call -- produces the final verdict. Unlike
DRUVA's per-symbol ``ask_as``, ARGUS reviews the whole candidate batch in one
call per persona (same rationale as ``pick_reviewer.review_picks``: a local
model is too slow for a call per symbol per persona).
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass

import structlog

from argus.advisor.digest import build_digest
from argus.advisor.llm import LLMBackend, LLMRequest
from argus.advisor.personas import Persona
from argus.advisor.pick_reviewer import (
    PickVerdict,
    Verdict,
    coerce_confidence,
    coerce_verdict,
    parse_picks_rows,
)
from argus.markets import Market
from argus.screener.base import Candidate

logger = structlog.get_logger(__name__)

_JSON_INSTRUCTIONS = (
    "Respond with strict JSON only, no markdown fencing and no prose outside "
    'the JSON. The JSON must be a single object of the form {"picks": '
    '[{"symbol": str, "verdict": "buy"|"watch"|"avoid", "confidence": int '
    '(0-100), "thesis": str}, ...]}, with exactly one entry per candidate '
    "listed below, using the given symbols."
)


@dataclass(frozen=True)
class CouncilVote:
    persona: str
    verdict: Verdict
    confidence: int
    thesis: str


@dataclass(frozen=True)
class CouncilVerdict:
    verdict: Verdict
    confidence: int
    votes: list[CouncilVote]
    summary: str


def _persona_system_prompt(persona: Persona) -> str:
    return (
        f"You are channelling {persona.name} ({persona.style}), reviewing "
        "candidates produced by a quantitative stock screener. Stay in "
        "character and be concise.\n\n"
        f"{persona.system_prompt}\n\n{_JSON_INSTRUCTIONS}"
    )


async def _review_as_persona(
    persona: Persona,
    candidates: list[Candidate],
    market: Market,
    backend: LLMBackend,
    valid_symbols: set[str],
) -> dict[str, CouncilVote]:
    """One batch call in ``persona``'s voice.

    Raises on backend failure or any other unexpected error -- the caller
    (``council_review``) fans these out via
    ``asyncio.gather(..., return_exceptions=True)`` so one persona's failure
    never takes the others down with it.
    """
    request = LLMRequest(
        system=_persona_system_prompt(persona), user=build_digest(candidates, market)
    )
    response = await backend.complete(request)
    if not response.text:
        return {}

    rows = parse_picks_rows(response.text)
    votes: dict[str, CouncilVote] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        symbol = row.get("symbol")
        if not isinstance(symbol, str) or symbol not in valid_symbols:
            continue
        votes[symbol] = CouncilVote(
            persona=persona.slug,
            verdict=coerce_verdict(row.get("verdict")),
            confidence=coerce_confidence(row.get("confidence")),
            thesis=str(row.get("thesis") or ""),
        )
    return votes


async def council_review(
    candidates: list[Candidate],
    market: Market,
    backend: LLMBackend,
    personas: list[Persona],
) -> dict[str, CouncilVerdict]:
    """Review ``candidates`` once per persona (all fanned out concurrently
    via ``asyncio.gather``) and fuse the results into one ``CouncilVerdict``
    per symbol that at least one persona voted on.

    Never raises: a per-persona backend failure (timeout, bad JSON, etc.) is
    logged and that persona simply contributes no votes -- fusion still runs
    over whichever personas responded, down to a single one. A symbol no
    persona voted on for gets no entry in the result at all.
    """
    if not candidates or not personas:
        return {}

    valid_symbols = {c.instrument.symbol for c in candidates}
    raw_results = await asyncio.gather(
        *(
            _review_as_persona(persona, candidates, market, backend, valid_symbols)
            for persona in personas
        ),
        return_exceptions=True,
    )

    votes_by_symbol: dict[str, list[tuple[Persona, CouncilVote]]] = defaultdict(list)
    for persona, result in zip(personas, raw_results, strict=True):
        if isinstance(result, BaseException):
            logger.warning(
                "advisor.council.persona_failed", persona=persona.slug, error=str(result)
            )
            continue
        for symbol, vote in result.items():
            votes_by_symbol[symbol].append((persona, vote))

    return {symbol: _fuse(votes) for symbol, votes in votes_by_symbol.items()}


def _fuse(votes: list[tuple[Persona, CouncilVote]]) -> CouncilVerdict:
    """Deterministic confidence-weighted vote fusion -- no LLM involved.

    Each vote contributes ``persona.weight * vote.confidence`` to its
    verdict class; the class with the highest total weight wins (ties break
    on ``buy`` > ``watch`` > ``avoid`` -- the dict literal's insertion
    order). Fused confidence is the winning class's *share* of total weight,
    scaled by the mean confidence of the voters who picked it -- so a
    narrow, low-conviction win scores lower than a lopsided, high-conviction
    one.
    """
    weight_by_verdict: dict[Verdict, float] = {"buy": 0.0, "watch": 0.0, "avoid": 0.0}
    for persona, vote in votes:
        weight_by_verdict[vote.verdict] += persona.weight * vote.confidence

    total_weight = sum(weight_by_verdict.values())
    winning_verdict = max(weight_by_verdict, key=lambda k: weight_by_verdict[k])
    winners = [(p, v) for p, v in votes if v.verdict == winning_verdict]
    mean_winner_confidence = sum(v.confidence for _, v in winners) / len(winners)

    if total_weight > 0:
        confidence = int(
            round(weight_by_verdict[winning_verdict] / total_weight * mean_winner_confidence)
        )
    else:
        confidence = 0
    confidence = max(0, min(100, confidence))

    return CouncilVerdict(
        verdict=winning_verdict,
        confidence=confidence,
        votes=[v for _, v in votes],
        summary=_summarize(votes, winning_verdict),
    )


def _summarize(votes: list[tuple[Persona, CouncilVote]], winning_verdict: Verdict) -> str:
    """Deterministic ``"2-1 buy (buffett 80, lynch 65 vs burry 40)"``-style summary."""
    winners = [(p, v) for p, v in votes if v.verdict == winning_verdict]
    dissenters = [(p, v) for p, v in votes if v.verdict != winning_verdict]
    header = f"{len(winners)}-{len(dissenters)} {winning_verdict}"
    winner_str = ", ".join(f"{p.slug} {v.confidence}" for p, v in winners)
    if not dissenters:
        return f"{header} ({winner_str})"
    dissent_str = ", ".join(f"{p.slug} {v.confidence}" for p, v in dissenters)
    return f"{header} ({winner_str} vs {dissent_str})"


def council_to_pick_verdicts(council: dict[str, CouncilVerdict]) -> dict[str, PickVerdict]:
    """Map council verdicts onto ``PickVerdict`` for reuse of the existing
    ``apply_verdicts``/persistence path.

    ``thesis`` is the deterministic vote summary plus the winning side's
    most confident persona's own thesis; ``risks`` is the most confident
    dissenter's thesis, or empty if the council was unanimous. The full
    per-persona vote breakdown rides along in ``votes`` for the UI.
    """
    result: dict[str, PickVerdict] = {}
    for symbol, verdict in council.items():
        winners = [v for v in verdict.votes if v.verdict == verdict.verdict]
        dissenters = [v for v in verdict.votes if v.verdict != verdict.verdict]

        thesis = verdict.summary
        if winners:
            top = max(winners, key=lambda v: v.confidence)
            if top.thesis:
                thesis = f"{verdict.summary} — {top.thesis}"

        risks = ""
        if dissenters:
            worst = max(dissenters, key=lambda v: v.confidence)
            risks = worst.thesis

        result[symbol] = PickVerdict(
            symbol=symbol,
            verdict=verdict.verdict,
            confidence=verdict.confidence,
            thesis=thesis,
            risks=risks,
            votes=tuple(
                {
                    "persona": v.persona,
                    "verdict": v.verdict,
                    "confidence": v.confidence,
                    "thesis": v.thesis,
                }
                for v in verdict.votes
            ),
        )
    return result
