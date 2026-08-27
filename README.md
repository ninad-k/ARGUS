# ARGUS

ARGUS is a daily stock screener for US and Indian markets. Multiple screening
strategies plus a local Ollama LLM suggest a handful of stocks each day using
technicals and fundamentals. All trading is simulated — **paper trading
only; no broker or order-execution code will ever exist in this codebase.**
Data sources are pluggable. Later phases add options intelligence and
orderflow analysis.

Status: **Phase 1 (foundation) — work in progress.**

## Stack

Python 3.12+, FastAPI + NiceGUI, SQLAlchemy 2.0 (async) + SQLite for the
control-plane database, DuckDB + Parquet for OHLCV bar data, APScheduler for
the daily post-close runs, httpx for data providers, pydantic v2 +
pydantic-settings for configuration, numpy/pandas for analysis.

## Dev setup

```bash
python3.12 -m venv .venv   # falls back to python3 if 3.12 isn't installed
source .venv/bin/activate
pip install -e ".[dev]"
```

Run the checks:

```bash
ruff check .
mypy argus
pytest
```

Configuration is environment-driven, prefixed `ARGUS_`, with per-domain
settings nested via a double underscore, e.g. `ARGUS_LLM__MODEL=llama3:8b`
or `ARGUS_PAPER__MAX_POSITIONS=25`. See `argus/config/` for the full list of
settings.

Runtime state (SQLite DB, DuckDB/Parquet OHLCV store) lives under
`~/.argus` by default; override with `ARGUS_DATA_DIR`.
