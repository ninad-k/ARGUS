"""Investor-persona council: fan-out collects one vote per responding persona,
weighted fusion matches hand-computed numbers exactly, a failing persona is
tolerated (others still produce a verdict), ``council_to_pick_verdicts``
mapping, and a ``PickVerdict.votes`` persistence round-trip.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select

from argus.advisor.council import (
    CouncilVerdict,
    CouncilVote,
    council_review,
    council_to_pick_verdicts,
)
from argus.advisor.llm import LLMBackend, LLMRequest, LLMResponse
from argus.advisor.personas import get_personas
from argus.advisor.pick_reviewer import PickVerdict
from argus.config import AppSettings
from argus.db import async_session, init_db
from argus.db.models import DailyPick
from argus.markets import US_NASDAQ, Instrument
from argus.screener.base import Candidate
from argus.screener.runner import ScreenResult, persist_screen_result

_PERSONA_SLUGS = ("buffett", "lynch", "burry")


class _RoutingBackend:
    """Test double: routes each call to a canned response by which persona's
    name appears in the request's system prompt (``council._persona_system_prompt``
    always leads with ``persona.name``), and can be told to fail for chosen
    personas."""

    provider = "static"
    model = "static-1"

    def __init__(
        self, responses_by_slug: dict[str, str], *, fail_slugs: frozenset[str] = frozenset()
    ) -> None:
        self._responses = responses_by_slug
        self._fail = fail_slugs
        self._personas_by_slug = {p.slug: p for p in get_personas(_PERSONA_SLUGS)}

    async def complete(self, req: LLMRequest) -> LLMResponse:
        for slug, persona in self._personas_by_slug.items():
            if persona.name in req.system:
                if slug in self._fail:
                    raise RuntimeError(f"simulated {slug} backend failure")
                return LLMResponse(
                    text=self._responses[slug], provider=self.provider, model=self.model
                )
        raise AssertionError(f"unroutable request system prompt: {req.system!r}")

    async def aclose(self) -> None:
        return None


def _picks_json(rows: list[tuple[str, str, int, str]]) -> str:
    """``rows`` of ``(symbol, verdict, confidence, thesis)`` -> the picks JSON text."""
    picks = [
        {"symbol": symbol, "verdict": verdict, "confidence": confidence, "thesis": thesis}
        for symbol, verdict, confidence, thesis in rows
    ]
    return json.dumps({"picks": picks})


def _candidates(symbols: list[str] = ["AAA", "BBB"]) -> list[Candidate]:  # noqa: B006
    return [
        Candidate(
            instrument=Instrument(symbol=sym, market_code=US_NASDAQ.code),
            strategy="momentum",
            score=80.0,
        )
        for sym in symbols
    ]


async def test_council_review_fans_out_and_collects_votes() -> None:
    backend: LLMBackend = _RoutingBackend(
        {
            "buffett": _picks_json(
                [("AAA", "buy", 80, "moat"), ("BBB", "watch", 50, "thin margins")]
            ),
            "lynch": _picks_json(
                [("AAA", "buy", 60, "growth"), ("BBB", "avoid", 40, "no earnings")]
            ),
            "burry": _picks_json(
                [("AAA", "avoid", 90, "leverage"), ("BBB", "avoid", 70, "cash burn")]
            ),
        }
    )
    personas = get_personas(_PERSONA_SLUGS)

    council = await council_review(_candidates(), US_NASDAQ, backend, personas)

    assert set(council) == {"AAA", "BBB"}
    assert {v.persona for v in council["AAA"].votes} == {"buffett", "lynch", "burry"}
    assert {v.persona for v in council["BBB"].votes} == {"buffett", "lynch", "burry"}


async def test_council_fusion_matches_hand_computed_weights() -> None:
    """buffett buy@80, lynch buy@60, burry avoid@90 (all weight=1.0):
    buy weight = 140, avoid weight = 90, total = 230 -> buy wins;
    confidence = round(140/230 * mean(80, 60)) = round(42.6087) = 43.
    """
    backend: LLMBackend = _RoutingBackend(
        {
            "buffett": _picks_json([("AAA", "buy", 80, "moat")]),
            "lynch": _picks_json([("AAA", "buy", 60, "growth")]),
            "burry": _picks_json([("AAA", "avoid", 90, "leverage")]),
        }
    )
    personas = get_personas(_PERSONA_SLUGS)

    council = await council_review(_candidates(["AAA"]), US_NASDAQ, backend, personas)

    verdict = council["AAA"]
    assert verdict.verdict == "buy"
    assert verdict.confidence == 43
    assert verdict.summary == "2-1 buy (buffett 80, lynch 60 vs burry 90)"


async def test_council_tolerates_a_failing_persona() -> None:
    backend: LLMBackend = _RoutingBackend(
        {
            "buffett": _picks_json([("AAA", "buy", 80, "moat")]),
            "lynch": _picks_json([("AAA", "buy", 60, "growth")]),
            "burry": _picks_json([("AAA", "avoid", 90, "leverage")]),
        },
        fail_slugs=frozenset({"burry"}),
    )
    personas = get_personas(_PERSONA_SLUGS)

    council = await council_review(_candidates(["AAA"]), US_NASDAQ, backend, personas)

    assert "AAA" in council
    verdict = council["AAA"]
    assert {v.persona for v in verdict.votes} == {"buffett", "lynch"}
    assert verdict.verdict == "buy"  # both surviving personas agree


async def test_council_review_no_candidates_or_personas_short_circuits() -> None:
    backend: LLMBackend = _RoutingBackend({})
    personas = get_personas(_PERSONA_SLUGS)

    assert await council_review([], US_NASDAQ, backend, personas) == {}
    assert await council_review(_candidates(["AAA"]), US_NASDAQ, backend, []) == {}


def test_council_to_pick_verdicts_mapping() -> None:
    votes = [
        CouncilVote(persona="buffett", verdict="buy", confidence=80, thesis="moat"),
        CouncilVote(persona="lynch", verdict="buy", confidence=60, thesis="growth"),
        CouncilVote(persona="burry", verdict="avoid", confidence=90, thesis="leverage"),
    ]
    council = {
        "AAA": CouncilVerdict(
            verdict="buy",
            confidence=43,
            votes=votes,
            summary="2-1 buy (buffett 80, lynch 60 vs burry 90)",
        )
    }

    result = council_to_pick_verdicts(council)

    assert set(result) == {"AAA"}
    pv = result["AAA"]
    assert pv.symbol == "AAA"
    assert pv.verdict == "buy"
    assert pv.confidence == 43
    # thesis = summary + the winning side's most confident persona's own thesis (buffett@80)
    assert pv.thesis == "2-1 buy (buffett 80, lynch 60 vs burry 90) — moat"
    # risks = the (only) dissenter's thesis
    assert pv.risks == "leverage"
    assert len(pv.votes) == 3
    assert {v["persona"] for v in pv.votes} == {"buffett", "lynch", "burry"}  # type: ignore[index]


def test_council_to_pick_verdicts_unanimous_has_no_risks() -> None:
    votes = [CouncilVote(persona="buffett", verdict="buy", confidence=80, thesis="moat")]
    council = {
        "AAA": CouncilVerdict(
            verdict="buy", confidence=80, votes=votes, summary="1-0 buy (buffett 80)"
        )
    }

    result = council_to_pick_verdicts(council)

    assert result["AAA"].risks == ""


async def test_pickverdict_votes_persist_round_trip(tmp_path: Path) -> None:
    settings = AppSettings(data_dir=tmp_path, _env_file=None)  # type: ignore[call-arg]
    await init_db(settings)

    verdict = PickVerdict(
        symbol="AAA",
        verdict="buy",
        confidence=43,
        thesis="council thesis",
        risks="council risk",
        votes=(
            {"persona": "buffett", "verdict": "buy", "confidence": 80, "thesis": "moat"},
            {"persona": "burry", "verdict": "avoid", "confidence": 90, "thesis": "leverage"},
        ),
    )
    candidate = Candidate(
        instrument=Instrument(symbol="AAA", market_code=US_NASDAQ.code),
        strategy="momentum",
        score=90.0,
        llm_verdict=verdict,
    )
    result = ScreenResult(
        market_code=US_NASDAQ.code,
        run_ts=datetime.now(UTC),
        universe_size=1,
        filtered_size=1,
        candidates=[candidate],
        top=[candidate],
    )

    run_id = await persist_screen_result(result, settings)

    async with async_session(settings) as session:
        pick = (
            (await session.execute(select(DailyPick).where(DailyPick.run_id == run_id)))
            .scalars()
            .one()
        )

    assert pick.llm_verdict_json is not None
    persisted_votes = pick.llm_verdict_json["votes"]
    assert persisted_votes == [
        {"persona": "buffett", "verdict": "buy", "confidence": 80, "thesis": "moat"},
        {"persona": "burry", "verdict": "avoid", "confidence": 90, "thesis": "leverage"},
    ]
