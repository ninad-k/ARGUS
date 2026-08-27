"""Rendering of a ``ScreenReport`` — the daily-picks doc humans read, in
Markdown and (Task 14) self-contained HTML.

Entry points: ``render_markdown_report``/``render_html_report`` (pure string
builders, used by tests and the smoke script), ``save_report`` (writes one
or both under ``{data_dir}/reports/``), and ``latest_report_path`` (finds
the newest saved report for the ``GET /api/v1/reports/latest`` endpoint).
"""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Literal

from argus.config import AppSettings, get_settings
from argus.orderflow.features import format_orderflow_summary
from argus.pipeline import ScreenReport
from argus.screener.base import Candidate

_NA = "-"

# Matches the filenames ``save_report`` writes: "{MARKET_CODE}_{YYYY-MM-DD}.{md,html}".
_REPORT_FILENAME_RE = re.compile(
    r"^(?P<market>[A-Za-z0-9_]+)_(?P<date>\d{4}-\d{2}-\d{2})\.(?P<ext>md|html)$"
)

_HTML_REPORT_CSS = """
body { font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif;
       margin: 2rem auto; max-width: 960px; padding: 0 1rem;
       background: #0e0e10; color: #e6e6e6; }
h1, h2, h3 { color: #ffffff; }
h1 { border-bottom: 2px solid #444; padding-bottom: 0.5rem; }
h2 { margin-top: 2rem; border-bottom: 1px solid #333; padding-bottom: 0.25rem; }
.meta p { margin: 0.2rem 0; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
th, td { border: 1px solid #333; padding: 0.4rem 0.6rem; text-align: left; font-size: 0.9rem; }
th { background: #1c1c1f; }
tr:nth-child(even) { background: #17171a; }
em { color: #999; }
"""


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


def _esc(value: object) -> str:
    return html.escape(str(value))


def render_html_report(report: ScreenReport) -> str:
    """Render ``report`` as a self-contained HTML document -- inline CSS,
    no external stylesheets/scripts/fonts/images -- for sharing outside a
    terminal or Markdown viewer. Mirrors ``render_markdown_report``'s
    structure section for section."""
    result = report.result
    run_ts = result.run_ts.strftime("%Y-%m-%d %H:%M:%S UTC")
    title = f"ARGUS Daily Picks — {_esc(result.market_code)}"

    parts: list[str] = [
        "<!doctype html>",
        '<html><head><meta charset="utf-8">',
        f"<title>{title}</title>",
        f"<style>{_HTML_REPORT_CSS}</style>",
        "</head><body>",
        f"<h1>{title}</h1>",
        '<div class="meta">',
        f"<p><strong>Run:</strong> {_esc(run_ts)}</p>",
        f"<p><strong>Universe:</strong> {result.universe_size} instruments scanned, "
        f"{result.filtered_size} passed filters, {len(result.candidates)} candidates</p>",
        f"<p><strong>Bars refreshed:</strong> {report.bars_refreshed}</p>",
        f"<p><strong>Symbols failed to refresh:</strong> {len(report.symbols_failed)}"
        + (f" ({_esc(', '.join(report.symbols_failed))})" if report.symbols_failed else "")
        + "</p>",
        f"<p><strong>LLM review used:</strong> {'yes' if report.llm_used else 'no'}</p>",
        f"<p><strong>Run id:</strong> {report.run_id}</p>",
        "</div>",
    ]

    if not result.top:
        parts.append("<p><em>No candidates cleared the screen today.</em></p>")
        parts.append("</body></html>")
        return "\n".join(parts)

    parts.append("<h2>Top Picks</h2>")
    parts.append(
        "<table><thead><tr><th>Symbol</th><th>Strategy</th><th>Score</th>"
        "<th>Stage</th><th>Entry</th><th>Stop</th><th>Target</th>"
        "<th>LLM Verdict</th><th>Confidence</th></tr></thead><tbody>"
    )
    for c in result.top:
        parts.append(
            "<tr>"
            f"<td>{_esc(c.instrument.symbol)}</td><td>{_esc(c.strategy)}</td>"
            f"<td>{c.score:.2f}</td><td>{_esc(c.stage or _NA)}</td>"
            f"<td>{_esc(_fmt_price(c.entry))}</td><td>{_esc(_fmt_price(c.stop))}</td>"
            f"<td>{_esc(_fmt_price(c.target))}</td><td>{_esc(_fmt_verdict(c))}</td>"
            f"<td>{_esc(_fmt_confidence(c))}</td>"
            "</tr>"
        )
    parts.append("</tbody></table>")

    parts.append("<h2>Pick Details</h2>")
    for c in result.top:
        parts.append(f"<h3>{_esc(c.instrument.symbol)}</h3>")
        parts.append(
            f"<p><strong>Strategy:</strong> {_esc(c.strategy)} | "
            f"<strong>Score:</strong> {c.score:.2f} | "
            f"<strong>Direction:</strong> {_esc(c.direction)} | "
            f"<strong>Stage:</strong> {_esc(c.stage or _NA)}</p>"
        )
        parts.append(f"<p><strong>Reason:</strong> {_esc(c.reason or _NA)}</p>")
        orderflow = c.features.get("orderflow")
        if isinstance(orderflow, dict):
            orderflow_line = format_orderflow_summary(orderflow)
            if orderflow_line:
                parts.append(f"<p><strong>Orderflow:</strong> {_esc(orderflow_line)}</p>")
        if c.llm_verdict is not None:
            parts.append(
                f"<p><strong>LLM Verdict:</strong> {_esc(c.llm_verdict.verdict)} "
                f"(confidence {c.llm_verdict.confidence})</p>"
            )
            parts.append(f"<p><strong>Thesis:</strong> {_esc(c.llm_verdict.thesis or _NA)}</p>")
            parts.append(f"<p><strong>Risks:</strong> {_esc(c.llm_verdict.risks or _NA)}</p>")

    if report.suggestions:
        parts.append("<h2>Derivative Ideas</h2>")
        parts.append(
            "<p><em>Analysis only -- these are not orders and nothing is ever "
            "placed automatically.</em></p>"
        )
        parts.append(
            "<table><thead><tr><th>Symbol</th><th>Type</th><th>Strike</th>"
            "<th>Expiry</th><th>Price</th><th>Delta</th><th>OI</th><th>Est. Cost</th>"
            "<th>Rationale</th></tr></thead><tbody>"
        )
        for symbol in (c.instrument.symbol for c in result.top):
            suggestion = report.suggestions.get(symbol)
            if suggestion is None:
                continue
            strike = f"{suggestion.strike:g}" if suggestion.strike is not None else _NA
            delta = f"{suggestion.delta:.2f}" if suggestion.delta is not None else _NA
            oi = f"{suggestion.oi:.0f}" if suggestion.oi is not None else _NA
            parts.append(
                "<tr>"
                f"<td>{_esc(suggestion.symbol)}</td><td>{_esc(suggestion.instrument_type)}</td>"
                f"<td>{_esc(strike)}</td><td>{_esc(suggestion.expiry.isoformat())}</td>"
                f"<td>{_esc(_fmt_price(suggestion.suggested_price))}</td><td>{_esc(delta)}</td>"
                f"<td>{_esc(oi)}</td><td>{_esc(_fmt_price(suggestion.est_cost))}</td>"
                f"<td>{_esc(suggestion.rationale)}</td>"
                "</tr>"
            )
        parts.append("</tbody></table>")

    parts.append("</body></html>")
    return "\n".join(parts)


def save_report(
    report: ScreenReport,
    out_dir: Path | None = None,
    *,
    fmt: Literal["md", "html", "both"] = "md",
) -> Path:
    """Render ``report`` and write it to ``{data_dir}/reports/{market}_{YYYY-MM-DD}.{md,html}``.

    ``out_dir`` overrides the ``reports`` directory (still named the same way
    within it) — used by the smoke script to write outside the default data
    directory. ``fmt`` selects which file(s) get written; the return value
    is always the Markdown path unless only the HTML one was written
    (``fmt="html"``), matching every existing caller's expectation of a
    single ``Path`` back for the primary (Markdown) report.
    """
    settings = get_settings()
    directory = out_dir if out_dir is not None else settings.data_dir / "reports"
    directory.mkdir(parents=True, exist_ok=True)

    date_str = report.result.run_ts.strftime("%Y-%m-%d")
    base = f"{report.result.market_code}_{date_str}"
    md_path = directory / f"{base}.md"
    html_path = directory / f"{base}.html"

    if fmt in ("md", "both"):
        md_path.write_text(render_markdown_report(report), encoding="utf-8")
    if fmt in ("html", "both"):
        html_path.write_text(render_html_report(report), encoding="utf-8")

    return html_path if fmt == "html" else md_path


def latest_report_path(
    market: str | None,
    *,
    settings: AppSettings | None = None,
    fmt: Literal["md", "html"] = "md",
    out_dir: Path | None = None,
) -> Path | None:
    """The newest saved report file for ``market`` (or across every market,
    when ``market`` is ``None``), by the date encoded in its filename --
    ``None`` if the reports directory doesn't exist or has no matching file.
    Backs ``GET /api/v1/reports/latest``.
    """
    resolved_settings = settings or get_settings()
    directory = out_dir if out_dir is not None else resolved_settings.data_dir / "reports"
    if not directory.exists():
        return None

    best: tuple[str, Path] | None = None
    for path in directory.glob(f"*.{fmt}"):
        match = _REPORT_FILENAME_RE.match(path.name)
        if match is None:
            continue
        if market is not None and match.group("market") != market:
            continue
        date_str = match.group("date")
        if best is None or date_str > best[0]:
            best = (date_str, path)

    return best[1] if best is not None else None
