# ARGUS

ARGUS is a daily stock screener for US and Indian markets. Multiple screening
strategies plus a local Ollama LLM suggest a handful of stocks each day using
technicals and fundamentals.

**ARGUS is analysis and paper-trading only.** All trading is simulated — no
broker connection, order-routing, or real trade-execution code exists in
this codebase, and none is planned. Nothing ARGUS does ever touches a real
brokerage account or places a real order.

Data sources are pluggable. Later phases add options intelligence and
orderflow analysis.

Status: **Phase 1 (foundation) — complete.**

## Stack

Python 3.12+, FastAPI + NiceGUI, SQLAlchemy 2.0 (async) + SQLite for the
control-plane database, DuckDB + Parquet for OHLCV bar data, APScheduler for
the daily post-close runs, httpx for data providers, pydantic v2 +
pydantic-settings for configuration, numpy/pandas for analysis.

## Quickstart

```bash
python3.12 -m venv .venv   # falls back to python3 if 3.12 isn't installed
source .venv/bin/activate
pip install -e ".[dev]"
```

Launch the web UI (REST API + NiceGUI dashboard, combined on one server):

```bash
argus
```

This starts on `http://127.0.0.1:8321` by default (override with
`ARGUS_UI__HOST` / `ARGUS_UI__PORT`) and prints the URL on startup. On first
run it creates the control-plane database, seeds a default `yfinance` data
source, and (unless disabled) starts the APScheduler post-close jobs.

- `/` — dashboard: per-market status + today's top picks across markets, with
  a "Run screen now" button per market
- `/picks` — full candidate table for any past run, with per-pick detail
  (reason, LLM thesis/risks, features)
- `/sources` — the data-sources admin screen: add/enable/disable/delete
  providers and test their connectivity
- `/settings` — read-only view of the active LLM/data/scheduler config, plus
  an Ollama connectivity check

The REST API backing all of this lives under `/api/v1` (`/api/v1/picks/latest`,
`/api/v1/runs`, `/api/v1/screen/run`, `/api/v1/sources`); interactive docs
are at `/docs`. `POST /api/v1/screen/run` runs the full daily pipeline
synchronously and can take from seconds to several minutes with live data —
there's no background task queue yet (Phase 2).

To run a single market's pipeline from the command line instead:

```bash
python scripts/smoke_daily.py --market US_NASDAQ           # offline, synthetic data
python scripts/smoke_daily.py --market US_NASDAQ --live     # real yfinance data
```

Run the checks:

```bash
ruff check .
mypy argus
pytest
```

Configuration is environment-driven, prefixed `ARGUS_`, with per-domain
settings nested via a double underscore, e.g. `ARGUS_LLM__MODEL=llama3:8b`,
`ARGUS_PAPER__MAX_POSITIONS=25`, or `ARGUS_UI__PORT=9000`. See `argus/config/`
for the full list of settings.

Runtime state (SQLite DB, DuckDB/Parquet OHLCV store) lives under
`~/.argus` by default; override with `ARGUS_DATA_DIR`.

## What Phase 1 does

- Universe construction, OHLCV bar refresh/caching (DuckDB + Parquet), and
  technical feature computation for the US (NYSE, NASDAQ) and India (NSE)
  markets
- Pluggable screener strategies (momentum, breakout) fused into ranked
  candidates, filtered through a default screening chain
- Optional single-pass LLM review of top candidates (local Ollama by
  default) producing a buy/watch/avoid verdict with thesis/risks
- A daily post-close scheduler (APScheduler) per market timezone, plus an
  on-demand REST/UI trigger
- Pluggable, DB-backed data sources (yfinance today; a `static` provider for
  tests/offline dev) with per-source health checks, manageable from the UI
  or the API
- A REST API and a NiceGUI web dashboard over all of the above
- Markdown daily-picks reports saved to disk

## Roadmap

- **Phase 2:** paper trading (simulated fills, positions, equity curve),
  TradingView integration, more data sources, webhooks
- **Phase 3:** options intelligence and orderflow analysis

ARGUS will never place a real trade. Every phase above stays within
simulated/paper trading and read-only market data/analysis — no broker
integration is planned at any point.
