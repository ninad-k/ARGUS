"""SQLAlchemy 2.0 declarative models for the ARGUS control-plane database.

This stores screener/paper-trading bookkeeping only. OHLCV bar data lives in
DuckDB/Parquet (a later task) — not here.
"""

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, ForeignKey, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class DataSource(Base):
    __tablename__ = "data_sources"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(unique=True)
    kind: Mapped[str]
    markets_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    config_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    priority: Mapped[int] = mapped_column(default=0)
    enabled: Mapped[bool] = mapped_column(default=True)
    last_health: Mapped[str | None] = mapped_column(default=None)
    created_at: Mapped[datetime]


class InstrumentRow(Base):
    __tablename__ = "instruments"
    __table_args__ = (UniqueConstraint("symbol", "market", name="uq_instrument_symbol_market"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str]
    market: Mapped[str]
    name: Mapped[str | None] = mapped_column(default=None)
    sector: Mapped[str | None] = mapped_column(default=None)
    lot_size: Mapped[int] = mapped_column(default=1)
    tick_size: Mapped[float] = mapped_column(default=0.01)
    has_options: Mapped[bool] = mapped_column(default=False)
    has_futures: Mapped[bool] = mapped_column(default=False)


class ScreenRun(Base):
    __tablename__ = "screen_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    market: Mapped[str]
    run_ts: Mapped[datetime]
    universe_size: Mapped[int]
    strategies_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    status: Mapped[str]
    duration_ms: Mapped[int | None] = mapped_column(default=None)


class DailyPick(Base):
    __tablename__ = "daily_picks"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("screen_runs.id"))
    symbol: Mapped[str]
    market: Mapped[str]
    strategy: Mapped[str]
    score: Mapped[float]
    stage: Mapped[str]
    reason: Mapped[str | None] = mapped_column(default=None)
    entry: Mapped[float | None] = mapped_column(default=None)
    stop: Mapped[float | None] = mapped_column(default=None)
    target: Mapped[float | None] = mapped_column(default=None)
    features_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    llm_verdict_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None)
    created_at: Mapped[datetime]


class OptionSuggestion(Base):
    __tablename__ = "option_suggestions"

    id: Mapped[int] = mapped_column(primary_key=True)
    pick_id: Mapped[int] = mapped_column(ForeignKey("daily_picks.id"))
    risk_level: Mapped[str]
    instrument_type: Mapped[str]
    strike: Mapped[float]
    expiry: Mapped[datetime]
    suggested_price: Mapped[float | None] = mapped_column(default=None)
    iv: Mapped[float | None] = mapped_column(default=None)
    delta: Mapped[float | None] = mapped_column(default=None)
    oi: Mapped[int | None] = mapped_column(default=None)
    rationale: Mapped[str | None] = mapped_column(default=None)


class PaperOrder(Base):
    """A simulated (paper) order. No broker/execution integration exists — ever."""

    __tablename__ = "paper_orders"

    id: Mapped[int] = mapped_column(primary_key=True)
    pick_id: Mapped[int | None] = mapped_column(ForeignKey("daily_picks.id"), default=None)
    symbol: Mapped[str]
    market: Mapped[str]
    side: Mapped[str]
    qty: Mapped[float]
    order_type: Mapped[str]
    status: Mapped[str]
    fill_price: Mapped[float | None] = mapped_column(default=None)
    slippage_bps: Mapped[float | None] = mapped_column(default=None)
    created_at: Mapped[datetime]
    filled_at: Mapped[datetime | None] = mapped_column(default=None)


class PaperPosition(Base):
    """A simulated (paper) position resulting from paper orders."""

    __tablename__ = "paper_positions"

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str]
    market: Mapped[str]
    qty: Mapped[float]
    avg_price: Mapped[float]
    opened_at: Mapped[datetime]
    closed_at: Mapped[datetime | None] = mapped_column(default=None)
    realized_pnl: Mapped[float | None] = mapped_column(default=None)


class PaperEquityPoint(Base):
    """Daily mark-to-market snapshot of the simulated paper portfolio."""

    __tablename__ = "paper_equity_points"

    id: Mapped[int] = mapped_column(primary_key=True)
    date: Mapped[datetime]
    market: Mapped[str]
    cash: Mapped[float]
    positions_value: Mapped[float]
    total_pnl: Mapped[float]


class WebhookEvent(Base):
    __tablename__ = "webhook_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    source: Mapped[str]
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    received_at: Mapped[datetime]
    processed: Mapped[bool] = mapped_column(default=False)


class LLMConfig(Base):
    __tablename__ = "llm_configs"

    id: Mapped[int] = mapped_column(primary_key=True)
    provider: Mapped[str]
    model: Mapped[str]
    base_url: Mapped[str]
    api_key: Mapped[str | None] = mapped_column(default=None)
    is_default: Mapped[bool] = mapped_column(default=False)
