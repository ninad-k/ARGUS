"""``render_markdown_report`` content and ``save_report`` file placement."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

from argus.advisor.pick_reviewer import PickVerdict
from argus.markets import US_NASDAQ, Instrument
from argus.options.suggester import DerivativeSuggestion, RiskLevel
from argus.pipeline import ScreenReport
from argus.reports import render_markdown_report, save_report
from argus.screener.base import Candidate
from argus.screener.runner import ScreenResult


def _report(
    *,
    top: list[Candidate] | None = None,
    candidates: list[Candidate] | None = None,
    suggestions: dict[str, DerivativeSuggestion] | None = None,
) -> ScreenReport:
    candidates = candidates if candidates is not None else (top or [])
    top = top if top is not None else candidates
    result = ScreenResult(
        market_code=US_NASDAQ.code,
        run_ts=datetime(2026, 8, 27, 16, 30, 0, tzinfo=UTC),
        universe_size=42,
        filtered_size=10,
        candidates=candidates,
        top=top,
    )
    return ScreenReport(
        result=result,
        run_id=7,
        bars_refreshed=123,
        symbols_failed=["BAD"],
        llm_used=True,
        suggestions=suggestions or {},
    )


def _suggestion(symbol: str) -> DerivativeSuggestion:
    return DerivativeSuggestion(
        symbol=symbol,
        market_code=US_NASDAQ.code,
        instrument_type="CE",
        strike=190.0,
        expiry=date(2026, 9, 25),
        suggested_price=12.40,
        iv=0.34,
        delta=0.42,
        oi=15000.0,
        lot_size=1,
        est_cost=12.40,
        rationale="MODERATE: NVDA 25 Sep 190C @ ~12.40, Δ0.42, OI 15.0k, IVR 34 — ATM call, 45 DTE",
        risk_level=RiskLevel.MODERATE,
    )


def _candidate(symbol: str, *, verdict: PickVerdict | None = None) -> Candidate:
    return Candidate(
        instrument=Instrument(symbol=symbol, market_code=US_NASDAQ.code),
        strategy="momentum+breakout",
        score=87.5,
        stage="breakout",
        reason="strong trend continuation",
        entry=150.0,
        stop=145.0,
        target=165.0,
        llm_verdict=verdict,
    )


def test_render_markdown_report_contains_symbols_scores_and_verdict() -> None:
    verdict = PickVerdict(
        symbol="AAPL", verdict="buy", confidence=85, thesis="strong momentum", risks="extended"
    )
    candidate = _candidate("AAPL", verdict=verdict)
    report = _report(top=[candidate])

    markdown = render_markdown_report(report)

    assert "AAPL" in markdown
    assert "87.50" in markdown
    assert "buy" in markdown
    assert "85" in markdown
    assert "strong momentum" in markdown
    assert "extended" in markdown
    assert US_NASDAQ.code in markdown
    assert "123" in markdown  # bars_refreshed
    assert "BAD" in markdown  # symbols_failed


def test_render_markdown_report_handles_no_verdict() -> None:
    candidate = _candidate("MSFT")
    report = _report(top=[candidate])

    markdown = render_markdown_report(report)

    assert "MSFT" in markdown
    expected_row = (
        "| MSFT | momentum+breakout | 87.50 | breakout | 150.00 | 145.00 | 165.00 | - | - |"
    )
    assert expected_row in markdown


def test_render_markdown_report_handles_no_candidates() -> None:
    report = _report(top=[])

    markdown = render_markdown_report(report)

    assert "No candidates cleared the screen today" in markdown


def test_render_markdown_report_renders_derivative_ideas_section() -> None:
    candidate = _candidate("NVDA")
    suggestion = _suggestion("NVDA")
    report = _report(top=[candidate], suggestions={"NVDA": suggestion})

    markdown = render_markdown_report(report)

    assert "## Derivative Ideas" in markdown
    assert "NVDA" in markdown
    assert "CE" in markdown
    assert "190" in markdown
    assert "2026-09-25" in markdown
    assert "12.40" in markdown
    assert "0.42" in markdown
    assert "15000" in markdown
    assert suggestion.rationale in markdown


def test_render_markdown_report_omits_derivative_ideas_section_when_empty() -> None:
    report = _report(top=[_candidate("MSFT")])

    markdown = render_markdown_report(report)

    assert "## Derivative Ideas" not in markdown


def test_save_report_writes_expected_path(tmp_path: Path) -> None:
    report = _report(top=[_candidate("AAPL")])

    path = save_report(report, out_dir=tmp_path)

    assert path == tmp_path / f"{US_NASDAQ.code}_2026-08-27.md"
    assert path.exists()
    assert "AAPL" in path.read_text(encoding="utf-8")


def test_save_report_creates_out_dir_if_missing(tmp_path: Path) -> None:
    target_dir = tmp_path / "nested" / "reports"
    report = _report(top=[_candidate("AAPL")])

    path = save_report(report, out_dir=target_dir)

    assert path.parent == target_dir
    assert path.exists()
