# ARGUS

ARGUS is a daily stock screener and paper-trading analysis tool for US
(NYSE, NASDAQ) and Indian (NSE) markets. Pluggable screening strategies,
technical + fundamental features, an approximated orderflow read, and an
optional local LLM review combine each post-close into a short list of
picks — which are then simulated through a paper-trading account so you can
see, honestly, whether they would have made money.

**ARGUS is analysis and paper-trading only.** All trading is simulated — no
broker connection, order-routing, or real trade-execution code exists in
this codebase, and none is planned. Nothing ARGUS does ever touches a real
brokerage account or places a real order. Every number on the `/paper` and
`/history` pages is a simulation against historical/incoming market data,
not real money.

Status: **Phases 1–3 complete.** Screening, paper trading, options
intelligence/orderflow, and pick-outcome review/attribution are all built
and covered by the test suite below.

## Architecture

- **`argus/markets/`** — market calendars (NYSE/NASDAQ/NSE trading days,
  holidays, timezones) and the `Instrument`/`Market` identity types.
- **`argus/data/`** — pluggable OHLCV price providers (`yfinance`,
  `tvscreener`, `nse`, `static`) behind one `PriceDataProvider` protocol,
  composed by `CompositePriceProvider` with per-source priority/health;
  fundamentals providers; universe construction; `data/store/duckdb_ohlcv.py`
  is the local DuckDB cache of daily + intraday bars.
- **`argus/indicators/`** — vectorized technical feature computation over
  OHLCV bars (numpy-backed).
- **`argus/orderflow/`** — OHLCV-derived approximations of orderflow
  concepts (gaps, liquidity zones, volume profile) — see the honest caveat
  below.
- **`argus/screener/`** — the `Strategy` ABC, the default filter chain, and
  `run_screen`'s universe → features → filters → strategies → fusion →
  ranking pipeline. Strategies: momentum, breakout, mean-reversion, value,
  orderflow-confluence.
- **`argus/advisor/`** — the LLM review layer: a single-pass reviewer or a
  multi-persona "council" (e.g. Buffett/Lynch/Druckenmiller-style votes),
  talking to a local Ollama backend by default.
- **`argus/options/`** — Black-Scholes analytics and a derivative-suggestion
  engine that proposes (never executes) an options idea per top pick.
- **`argus/paper/`** — the paper-trading engine: simulated order queuing,
  next-open fills with slippage, stop/target exit rules, and cash/equity
  bookkeeping. No broker/execution integration exists anywhere in this
  package.
- **`argus/analysis/`** — read-only analytics over historical picks and the
  paper account: `outcomes.py` walks each pick's bars forward to classify
  it hit-target/hit-stop/expired/open; `attribution.py` joins paper fills
  back to their originating picks to answer "which strategies and LLM
  verdicts actually made (simulated) money."
- **`argus/pipeline.py`** — `run_daily_pipeline`, the single entry point
  tying every layer above together for one market.
- **`argus/jobs/scheduler.py`** — APScheduler cron jobs firing at each
  market's own post-close time, in its own timezone.
- **`argus/reports.py`** — Markdown + self-contained HTML daily-picks
  reports, saved to disk.
- **`argus/api/`** — the versioned REST API (FastAPI) over all of the above.
- **`argus/ui/`** — the NiceGUI web dashboard (dashboard/picks/paper/
  history/sources/settings pages), mounted on the same FastAPI app.
- **`argus/db/`** — SQLAlchemy 2.0 async models/session for the SQLite
  control-plane database (runs, picks, orders, positions, sources, ...).

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

To run a single market's pipeline from the command line instead of waiting
for the scheduler:

```bash
python scripts/smoke_daily.py --market US_NASDAQ                    # offline, synthetic data
python scripts/smoke_daily.py --market US_NASDAQ --live              # real yfinance data
python scripts/smoke_daily.py --market US_NASDAQ --live --paper      # also run the paper cycle
python scripts/smoke_daily.py --market US_NASDAQ --no-llm            # skip the LLM review
```

Run the checks:

```bash
ruff check .
mypy argus
pytest
```

(435+ tests pass offline; a handful marked `network` are deselected by
default and only exercise live provider calls.)

## Pages tour

- **`/` (Dashboard)** — per-market status cards (last run, universe size,
  pick count) with a "Run screen now" button per market, plus today's top
  picks merged across every market.
- **`/picks`** — full candidate table for any past run, with per-pick
  detail: reason, LLM thesis/risks (and per-persona council votes, if
  council review is enabled), computed features, orderflow summary, and any
  attached derivative-idea suggestion.
- **`/paper`** — the simulated portfolio: cash per currency domain (USD for
  US markets, INR for NSE), open positions with live unrealized P&L, recent
  orders, and an equity curve chart. A guarded "Reset Account" button wipes
  every paper order/position/cash/equity row.
- **`/history`** — pick-outcome review and paper-vs-pick attribution.
  Summary cards (hit rate, stop rate, expectancy, avg winner/loser %), a
  per-strategy breakdown table, a color-coded outcomes table (did each pick
  hit its target, hit its stop, expire, or is it still open — with max
  favorable/adverse excursion), and an attribution section answering "which
  strategies and LLM verdicts actually made the paper account money" (P&L
  by strategy, P&L by verdict, per-position detail).
- **`/sources`** — the data-sources admin screen: add/enable/disable/delete
  providers and test their connectivity.
- **`/settings`** — read-only view of the active LLM/data/options/webhook/
  scheduler config, plus an Ollama connectivity check and next-run times.

The REST API backing all of this lives under `/api/v1`
(`/api/v1/picks/latest`, `/api/v1/runs`, `/api/v1/screen/run`,
`/api/v1/sources`, `/api/v1/paper/*`, `/api/v1/history/outcomes`,
`/api/v1/history/attribution`, `/api/v1/reports/latest`,
`/api/v1/webhooks/tradingview/<token>`); interactive docs are at `/docs`.
`POST /api/v1/screen/run` runs the full daily pipeline synchronously and can
take from seconds to several minutes with live data — there's no background
task queue.

## How a daily run works

1. **Scheduler fires** (`argus/jobs/scheduler.py`) at each market's
   post-close time in its own timezone — or you trigger one manually via
   the dashboard button, the smoke script, or `POST /api/v1/screen/run`.
2. **Universe + bars.** The configured universe provider resolves that
   market's instrument list; new OHLCV bars are refreshed into the DuckDB
   cache with bounded concurrency and per-symbol failure tolerance (one bad
   symbol never aborts the run).
3. **Screen.** Every applicable strategy runs over the filtered universe,
   producing ranked `Candidate`s; candidates picked by multiple strategies
   are fused into one (e.g. `"momentum+breakout"`) with a combined score.
4. **Orderflow + options annotation.** Top picks get intraday bars
   refreshed and an approximated `OrderflowFeatures` read attached; if
   options are enabled, a derivative-idea suggestion is generated per pick.
5. **LLM review** (optional). The top picks are sent to a local LLM — either
   a single reviewer call or a multi-persona council — for a buy/watch/avoid
   verdict with thesis/risks.
6. **Persist.** The run and its picks are written to the control-plane DB
   (`ScreenRun`, `DailyPick`, `OptionSuggestion`).
7. **Paper cycle.** Yesterday's queued orders fill against today's opening
   bars (with slippage); stop/target exit rules are applied to open
   positions; new orders are queued from today's picks (long-only, no
   short/margin simulation); the day's equity is snapshotted.
8. **Report saved.** A Markdown + HTML daily-picks report is written to
   `{data_dir}/reports/`.

Later, `/history` walks each pick's subsequent bars forward to see whether
it actually hit its target or stop, and joins the paper account's fills back
to the picks and LLM verdicts that produced them — turning the whole loop
into a feedback signal instead of a one-way broadcast.

## Configuration reference

Configuration is environment-driven, prefixed `ARGUS_`, with per-domain
settings nested via a double underscore, e.g. `ARGUS_LLM__MODEL=llama3:8b`.
The full set of defaults lives in `argus/config/`; the most commonly
overridden ones:

| Variable | Default | What it does |
|---|---|---|
| `ARGUS_DATA_DIR` | `~/.argus` | Where the SQLite DB, DuckDB store, and saved reports live. |
| `ARGUS_UI__HOST` / `ARGUS_UI__PORT` | `127.0.0.1` / `8321` | Combined API+UI server bind address. |
| `ARGUS_LLM_ENABLED` | `true` | Turns the LLM review step on/off. |
| `ARGUS_LLM_PROVIDER` / `ARGUS_LLM_MODEL` | `ollama` / `gemma3:4b` | LLM backend + model. |
| `ARGUS_LLM_BASE_URL` | `http://localhost:11434` | Local Ollama server URL. |
| `ARGUS_LLM_COUNCIL_ENABLED` | `false` | Multi-persona council review instead of a single pass. |
| `ARGUS_LLM_COUNCIL_PERSONAS` | `buffett,lynch,druckenmiller` | Comma-separated persona slugs when the council is on. |
| `ARGUS_DATA_UNIVERSE_SIZE_PER_MARKET` | `300` | Instruments scanned per market. |
| `ARGUS_DATA_BAR_LOOKBACK_DAYS` | `400` | Daily-bar history window pulled per symbol. |
| `ARGUS_OPTIONS_ENABLED` | `true` | Turns derivative-idea suggestions on/off. |
| `ARGUS_OPTIONS_RISK_LEVEL` | `moderate` | Default suggester risk profile (`conservative`/`moderate`/`aggressive`). |
| `ARGUS_PAPER_STARTING_CASH_US` / `ARGUS_PAPER_STARTING_CASH_INDIA` | `100000` / `1000000` | Simulated starting cash per currency domain. |
| `ARGUS_PAPER_POSITION_SIZE_PCT` | `5.0` | % of domain equity sized into each new paper position. |
| `ARGUS_PAPER_MAX_POSITIONS` | `10` | Max concurrent open paper positions. |
| `ARGUS_PAPER_SLIPPAGE_BPS` | `5` | Simulated slippage applied to every fill. |
| `ARGUS_SCHEDULER_ENABLED` | `true` | Turns the post-close cron jobs on/off. |
| `ARGUS_SCHEDULER_US_POST_CLOSE` / `ARGUS_SCHEDULER_INDIA_POST_CLOSE` | `16:30` / `18:30` | Local post-close fire times. |
| `ARGUS_WEBHOOKS_TRADINGVIEW_TOKEN` | *(unset)* | Enables the TradingView alert webhook (see below) when set. |

## Data sources

Every price/fundamentals source implements one `PriceDataProvider` /
`FundamentalsProvider` protocol and is registered in the control-plane DB
(manageable from `/sources` or the `/api/v1/sources` API), so multiple
sources can be composed with priority + health-check fallback:

- **`yfinance`** — the default: free, no API key, decent history depth for
  both US and NSE tickers.
- **`tvscreener`** — TradingView's screener API, used for universe
  construction/scanning where configured.
- **`nse`** — a direct NSE India HTTP data source for NSE-specific
  instruments/quotes.
- **`static`** — an in-memory/seeded provider for tests and fully offline
  runs (used by `scripts/smoke_daily.py`'s default mode).

**TradingView alert webhook.** Set `ARGUS_WEBHOOKS_TRADINGVIEW_TOKEN` to a
random secret, then point a TradingView alert's "Webhook URL" field at
`https://<your-argus-host>/api/v1/webhooks/tradingview/<token>`. The alert
body (JSON or plain text — both accepted) is stored as a `WebhookEvent` row
and listed at `GET /api/v1/webhooks/events`; turning a webhook event into a
screener signal is not wired up yet. An empty/unset token disables the
endpoint entirely (every request 404s).

## A note on "orderflow"

`argus/orderflow/` computes gap, liquidity-zone, and volume-profile features
**from OHLCV bars alone** — daily bars, and intraday bars where a provider
has them (currently only `yfinance`). This is a genuinely useful
approximation of orderflow *concepts* (where volume has clustered, where
gaps sit unfilled, where liquidity may pool), but it is **not** real
orderflow: there is no tape, no order book, no bid/ask imbalance, and no
Level 2 data anywhere in this codebase. Treat every orderflow feature as a
volume/price-derived heuristic, not a market-microstructure read.

## ARGUS will never place a real trade

Every phase of this project stays within simulated/paper trading and
read-only market data/analysis. There is no broker integration, no
order-routing code, and no path by which a `DailyPick`, a paper order, or a
derivative suggestion turns into a real transaction — none is planned at
any point.
