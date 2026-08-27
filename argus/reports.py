"""Markdown rendering of a ``ScreenReport`` — the daily-picks doc humans read.

Two entry points: ``render_markdown_report`` (pure string builder, used by
tests and the smoke script) and ``save_report`` (writes it under
``{data_dir}/reports/``).
"""

from __future__ import annotations

from pathlib import Path

from argus.config import get_settings
from argus.orderflow.features import format_orderflow_summary
from argus.pipeline import ScreenReport
from argus.screener.base import Candidate

_NA = "-"


def _fmt_price(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else _NA


def _fmt_verdict(candidate: Candidate) -> str:
    v = candidate.llm_verdict
    return v.verdict if v is not None else _NA


def _fmt_confidence(candidate: Candidate) -> str:
    v = candidate.llm_verdict
    return str(v.confidence) if v is not None else _NA


def render_markdown_report(report: ScreenReport) -> str:
    """Render ``report`` as a readable Markdown daily-picks document."""
    result = report.result
    run_ts = result.run_ts.strftime("%Y-%m-%d %H:%M:%S UTC")

    lines: list[str] = [
        f"# ARGUS Daily Picks — {result.market_code}",
        "",
        f"**Run:** {run_ts}  ",
        f"**Universe:** {result.universe_size} instruments scanned, "
        f"{result.filtered_size} passed filters, {len(result.candidates)} candidates  ",
        f"**Bars refreshed:** {report.bars_refreshed}  ",
        f"**Symbols failed to refresh:** {len(report.symbols_failed)}"
        + (f" ({', '.join(report.symbols_failed)})" if report.symbols_failed else "")
        + "  ",
        f"**LLM review used:** {'yes' if report.llm_used else 'no'}  ",
        f"**Run id:** {report.run_id}",
        "",
    ]

    if not result.top:
        lines.append("_No candidates cleared the screen today._")
        return "\n".join(lines) + "\n"

    lines.append("## Top Picks")
    lines.append("")
    lines.append(
        "| Symbol | Strategy | Score | Stage | Entry | Stop | Target | "
        "LLM Verdict | Confidence |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for c in result.top:
        lines.append(
            f"| {c.instrument.symbol} | {c.strategy} | {c.score:.2f} | {c.stage or _NA} | "
            f"{_fmt_price(c.entry)} | {_fmt_price(c.stop)} | {_fmt_price(c.target)} | "
            f"{_fmt_verdict(c)} | {_fmt_confidence(c)} |"
        )
    lines.append("")

    lines.append("## Pick Details")
    lines.append("")
    for c in result.top:
        lines.append(f"### {c.instrument.symbol}")
        lines.append("")
        lines.append(
            f"**Strategy:** {c.strategy} | **Score:** {c.score:.2f} | "
            f"**Direction:** {c.direction} | **Stage:** {c.stage or _NA}"
        )
        lines.append("")
        lines.append(f"**Reason:** {c.reason or _NA}")
        orderflow = c.features.get("orderflow")
        if isinstance(orderflow, dict):
            orderflow_line = format_orderflow_summary(orderflow)
            if orderflow_line:
                lines.append(f"**Orderflow:** {orderflow_line}")
        if c.llm_verdict is not None:
            lines.append("")
            lines.append(
                f"**LLM Verdict:** {c.llm_verdict.verdict} "
                f"(confidence {c.llm_verdict.confidence})"
            )
            lines.append(f"**Thesis:** {c.llm_verdict.thesis or _NA}")
            lines.append(f"**Risks:** {c.llm_verdict.risks or _NA}")
        lines.append("")

    if report.suggestions:
        lines.append("## Derivative Ideas")
        lines.append("")
        lines.append(
            "_Analysis only -- these are not orders and nothing is ever placed "
            "automatically._"
        )
        lines.append("")
        lines.append("| Symbol | Type | Strike | Expiry | Price | Δ | OI | Est. Cost | Rationale |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for symbol in (c.instrument.symbol for c in result.top):
            suggestion = report.suggestions.get(symbol)
            if suggestion is None:
                continue
            strike = f"{suggestion.strike:g}" if suggestion.strike is not None else _NA
            delta = f"{suggestion.delta:.2f}" if suggestion.delta is not None else _NA
            oi = f"{suggestion.oi:.0f}" if suggestion.oi is not None else _NA
            lines.append(
                f"| {suggestion.symbol} | {suggestion.instrument_type} | {strike} | "
                f"{suggestion.expiry.isoformat()} | {_fmt_price(suggestion.suggested_price)} | "
                f"{delta} | {oi} | {_fmt_price(suggestion.est_cost)} | {suggestion.rationale} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def save_report(report: ScreenReport, out_dir: Path | None = None) -> Path:
    """Render ``report`` and write it to ``{data_dir}/reports/{market}_{YYYY-MM-DD}.md``.

    ``out_dir`` overrides the ``reports`` directory (still named the same way
    within it) — used by the smoke script to write outside the default data
    directory.
    """
    settings = get_settings()
    directory = out_dir if out_dir is not None else settings.data_dir / "reports"
    directory.mkdir(parents=True, exist_ok=True)

    date_str = report.result.run_ts.strftime("%Y-%m-%d")
    path = directory / f"{report.result.market_code}_{date_str}.md"
    path.write_text(render_markdown_report(report), encoding="utf-8")
    return path
