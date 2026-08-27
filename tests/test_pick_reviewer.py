"""``review_picks``/``apply_verdicts`` against a ``StaticBackend`` test double,
plus a ``persist_screen_result`` round-trip of ``llm_verdict_json``.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select

from argus.advisor.llm import LLMBackend, LLMRequest, LLMResponse
from argus.advisor.pick_reviewer import PickVerdict, apply_verdicts, review_picks
from argus.config import AppSettings
from argus.db import async_session, init_db
from argus.db.models import DailyPick
from argus.markets import US_NASDAQ, Instrument
from argus.screener.base import Candidate
from argus.screener.runner import ScreenResult, persist_screen_result


class StaticBackend:
    """Test double returning a canned response, or raising, per construction."""

    provider = "static"
    model = "static-1"

    def __init__(self, text: str = "", *, error: Exception | None = None) -> None:
        self._text = text
        self._error = error

    async def complete(self, req: LLMRequest) -> LLMResponse:
        if self._error is not None:
            raise self._error
        return LLMResponse(text=self._text, provider=self.provider, model=self.model)

    async def aclose(self) -> None:
        return None


def _static_backend(text: str = "", *, error: Exception | None = None) -> LLMBackend:
    return StaticBackend(text=text, error=error)


def _candidates() -> list[Candidate]:
    return [
        Candidate(
            instrument=Instrument(symbol="AAA", market_code=US_NASDAQ.code),
            strategy="momentum",
            score=80.0,
            features={"close": 100.0, "rsi_14": 55.0},
        ),
        Candidate(
            instrument=Instrument(symbol="BBB", market_code=US_NASDAQ.code),
            strategy="breakout",
            score=70.0,
            features={"close": float("nan")},
        ),
    ]


_CANNED_JSON = """Here you go:
```json
{"picks": [
    {"symbol": "AAA", "verdict": "buy", "confidence": 150, "thesis": "strong", "risks": "extended"},
    {"symbol": "BBB", "verdict": "hold", "confidence": 40, "thesis": "meh", "risks": "choppy"},
    {"symbol": "ZZZ", "verdict": "avoid", "confidence": 10, "thesis": "n/a", "risks": "n/a"}
]}
```"""


async def test_review_picks_parses_clamps_and_drops_unknown_symbols() -> None:
    verdicts = await review_picks(_candidates(), US_NASDAQ, _static_backend(_CANNED_JSON))

    assert set(verdicts) == {"AAA", "BBB"}  # ZZZ isn't one of our candidates -> dropped

    aaa = verdicts["AAA"]
    assert aaa.verdict == "buy"
    assert aaa.confidence == 100  # clamped from 150

    bbb = verdicts["BBB"]
    assert bbb.verdict == "watch"  # "hold" isn't a recognized verdict -> defaults to watch
    assert bbb.confidence == 40


async def test_review_picks_no_candidates_short_circuits() -> None:
    verdicts = await review_picks([], US_NASDAQ, _static_backend(_CANNED_JSON))
    assert verdicts == {}


async def test_review_picks_backend_failure_returns_empty() -> None:
    backend = _static_backend(error=TimeoutError("ollama is down"))
    verdicts = await review_picks(_candidates(), US_NASDAQ, backend)
    assert verdicts == {}


async def test_review_picks_empty_response_returns_empty() -> None:
    verdicts = await review_picks(_candidates(), US_NASDAQ, _static_backend(""))
    assert verdicts == {}


async def test_review_picks_unparseable_response_returns_empty() -> None:
    verdicts = await review_picks(_candidates(), US_NASDAQ, _static_backend("not json at all"))
    assert verdicts == {}


async def test_apply_verdicts_annotates_matching_candidates_only() -> None:
    candidates = _candidates()
    result = ScreenResult(
        market_code=US_NASDAQ.code,
        run_ts=datetime.now(UTC),
        universe_size=2,
        filtered_size=2,
        candidates=candidates,
        top=candidates[:1],
    )
    verdicts = await review_picks(candidates, US_NASDAQ, _static_backend(_CANNED_JSON))

    apply_verdicts(result, verdicts)

    assert result.candidates[0].llm_verdict == verdicts["AAA"]
    assert result.candidates[1].llm_verdict == verdicts["BBB"]
    # `top` is a slice of the same Candidate objects, so it sees the mutation too.
    assert result.top[0].llm_verdict == verdicts["AAA"]


async def test_apply_verdicts_leaves_unmatched_candidates_none() -> None:
    candidates = _candidates()
    result = ScreenResult(
        market_code=US_NASDAQ.code,
        run_ts=datetime.now(UTC),
        universe_size=2,
        filtered_size=2,
        candidates=candidates,
        top=candidates,
    )
    apply_verdicts(result, {})
    assert all(c.llm_verdict is None for c in result.candidates)


async def test_persist_screen_result_round_trips_llm_verdict_json(tmp_path: Path) -> None:
    settings = AppSettings(data_dir=tmp_path, _env_file=None)  # type: ignore[call-arg]
    await init_db(settings)

    verdict = PickVerdict(
        symbol="AAA", verdict="buy", confidence=85, thesis="strong trend", risks="extended"
    )
    candidate = Candidate(
        instrument=Instrument(symbol="AAA", market_code=US_NASDAQ.code),
        strategy="momentum",
        score=90.0,
        llm_verdict=verdict,
    )
    no_verdict_candidate = Candidate(
        instrument=Instrument(symbol="BBB", market_code=US_NASDAQ.code),
        strategy="breakout",
        score=60.0,
    )
    result = ScreenResult(
        market_code=US_NASDAQ.code,
        run_ts=datetime.now(UTC),
        universe_size=2,
        filtered_size=2,
        candidates=[candidate, no_verdict_candidate],
        top=[candidate, no_verdict_candidate],
    )

    run_id = await persist_screen_result(result, settings)

    async with async_session(settings) as session:
        picks = (
            (await session.execute(select(DailyPick).where(DailyPick.run_id == run_id)))
            .scalars()
            .all()
        )
    by_symbol = {p.symbol: p for p in picks}

    assert by_symbol["AAA"].llm_verdict_json == asdict(verdict)
    assert by_symbol["BBB"].llm_verdict_json is None
