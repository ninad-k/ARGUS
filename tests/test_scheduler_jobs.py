"""``build_scheduler`` job/cron wiring and ``run_market_job``'s holiday-skip /
exception-containment behavior.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from apscheduler.triggers.cron import CronTrigger

from argus.config import AppSettings
from argus.jobs.scheduler import build_scheduler, run_market_job
from argus.markets import IN_NSE, US_NASDAQ, US_NYSE, get_market
from argus.pipeline import ScreenReport
from argus.screener.runner import ScreenResult


def _settings(tmp_path: Path) -> AppSettings:
    return AppSettings(data_dir=tmp_path, _env_file=None)  # type: ignore[call-arg]


def _field(trigger: CronTrigger, name: str) -> Any:
    return next(f for f in trigger.fields if f.name == name)


def test_build_scheduler_returns_two_jobs_with_correct_cron_and_timezone(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    scheduler = build_scheduler(settings)

    jobs = {job.id: job for job in scheduler.get_jobs()}
    assert set(jobs) == {"us_post_close", "india_post_close"}

    us_trigger = jobs["us_post_close"].trigger
    assert isinstance(us_trigger, CronTrigger)
    assert str(_field(us_trigger, "hour")) == "16"
    assert str(_field(us_trigger, "minute")) == "30"
    assert str(_field(us_trigger, "day_of_week")) == "mon-fri"
    assert str(us_trigger.timezone) == settings.scheduler.us_timezone

    india_trigger = jobs["india_post_close"].trigger
    assert isinstance(india_trigger, CronTrigger)
    assert str(_field(india_trigger, "hour")) == "18"
    assert str(_field(india_trigger, "minute")) == "30"
    assert str(_field(india_trigger, "day_of_week")) == "mon-fri"
    assert str(india_trigger.timezone) == settings.scheduler.india_timezone


def test_build_scheduler_respects_custom_post_close_times(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.scheduler.us_post_close = "17:05"
    scheduler = build_scheduler(settings)

    jobs = {job.id: job for job in scheduler.get_jobs()}
    us_trigger = jobs["us_post_close"].trigger
    assert isinstance(us_trigger, CronTrigger)
    assert str(_field(us_trigger, "hour")) == "17"
    assert str(_field(us_trigger, "minute")) == "5"


async def test_run_market_job_skips_on_holiday(monkeypatch: pytest.MonkeyPatch) -> None:
    # Computed the same way run_market_job computes "today" -- via the
    # market's own timezone -- so this can't flake around a UTC day boundary.
    today_in_market_tz = datetime.now(US_NASDAQ.timezone).date()
    holiday_market = replace(US_NASDAQ, holidays=frozenset({today_in_market_tz}))

    def _fake_get_market(code: str) -> Any:
        assert code == US_NASDAQ.code
        return holiday_market

    monkeypatch.setattr("argus.jobs.scheduler.get_market", _fake_get_market)

    called = False

    async def _fail_if_called(*args: Any, **kwargs: Any) -> ScreenReport:
        nonlocal called
        called = True
        raise AssertionError("pipeline should not run on a holiday")

    monkeypatch.setattr("argus.jobs.scheduler.run_daily_pipeline", _fail_if_called)

    report = await run_market_job(US_NASDAQ.code)

    assert report is None
    assert called is False


async def test_run_market_job_swallows_pipeline_exceptions(monkeypatch: pytest.MonkeyPatch) -> None:
    # Use the real market so is_trading_day only skips on an actual weekend/holiday;
    # if today happens to not be a trading day the pipeline path below wouldn't run
    # at all, so pin a definitely-open Monday via a trading-everyday fake market.
    always_open_market = replace(get_market(US_NASDAQ.code), holidays=frozenset())
    monkeypatch.setattr("argus.jobs.scheduler.get_market", lambda code: always_open_market)

    async def _raise(*args: Any, **kwargs: Any) -> ScreenReport:
        raise RuntimeError("pipeline blew up")

    monkeypatch.setattr("argus.jobs.scheduler.run_daily_pipeline", _raise)

    report = await run_market_job(US_NASDAQ.code)

    assert report is None


async def test_run_market_job_saves_report_on_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    always_open_market = replace(get_market(IN_NSE.code), holidays=frozenset())
    monkeypatch.setattr("argus.jobs.scheduler.get_market", lambda code: always_open_market)

    fake_result = ScreenResult(
        market_code=IN_NSE.code,
        run_ts=datetime.now(UTC),
        universe_size=1,
        filtered_size=1,
        candidates=[],
        top=[],
    )
    fake_report = ScreenReport(
        result=fake_result, run_id=1, bars_refreshed=0, symbols_failed=[], llm_used=False
    )

    async def _fake_pipeline(*args: Any, **kwargs: Any) -> ScreenReport:
        return fake_report

    monkeypatch.setattr("argus.jobs.scheduler.run_daily_pipeline", _fake_pipeline)

    saved: dict[str, Any] = {}

    def _fake_save_report(report: ScreenReport, out_dir: Path | None = None) -> Path:
        saved["report"] = report
        return tmp_path / "fake_report.md"

    monkeypatch.setattr("argus.jobs.scheduler.save_report", _fake_save_report)

    report = await run_market_job(IN_NSE.code)

    assert report is fake_report
    assert saved["report"] is fake_report


def test_market_registry_unaffected_by_dataclasses_replace() -> None:
    # `replace()` in the tests above must not mutate the shared registry singletons.
    assert US_NASDAQ.holidays != frozenset()
    assert US_NYSE.holidays != frozenset()
