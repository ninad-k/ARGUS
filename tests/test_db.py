"""init_db() creates tables and a DataSource row round-trips."""

from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select

from argus.config import AppSettings
from argus.db.models import DataSource
from argus.db.session import async_session, init_db


async def test_init_db_creates_tables_and_round_trips_a_row(tmp_path: Path) -> None:
    settings = AppSettings(data_dir=tmp_path, _env_file=None)  # type: ignore[call-arg]
    await init_db(settings)

    async with async_session(settings) as session:
        session.add(
            DataSource(
                name="yfinance",
                kind="ohlcv",
                markets_json={"markets": ["US_NYSE"]},
                config_json={},
                priority=1,
                enabled=True,
                last_health=None,
                created_at=datetime.now(UTC),
            )
        )
        await session.commit()

    async with async_session(settings) as session:
        result = await session.execute(select(DataSource).where(DataSource.name == "yfinance"))
        row = result.scalar_one()
        assert row.kind == "ohlcv"
        assert row.markets_json == {"markets": ["US_NYSE"]}
        assert row.enabled is True

    assert (tmp_path / "argus.db").exists()
