"""APScheduler wiring for the daily post-close screening jobs.

Two cron triggers, one per market group, fire at each market's own
post-close time in its own timezone (``SchedulerSettings``):

- US post-close: NASDAQ then NYSE, sequentially, Mon-Fri in ``us_timezone``.
- India post-close: NSE, Mon-Fri in ``india_timezone``.

``run_market_job`` is the plain, directly-testable job body: it skips
holidays, never lets a pipeline failure escape (a scheduler must keep
running even if one day's run blows up), and saves the Markdown report after
a successful run.
"""

from __future__ import annotations

from datetime import datetime

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from argus.config import AppSettings, get_settings
from argus.data.store.duckdb_ohlcv import BarStore
from argus.markets import IN_NSE, US_NASDAQ, US_NYSE, get_market
from argus.paper.engine import run_paper_cycle
from argus.pipeline import ScreenReport, run_daily_pipeline
from argus.reports import save_report

logger = structlog.get_logger(__name__)


async def run_market_job(market_code: str) -> ScreenReport | None:
    """Run the daily pipeline for ``market_code`` and save its report.

    Returns ``None`` (without running the pipeline) on a non-trading day for
    that market, or if the pipeline itself raises -- callers (the scheduler)
    must never see an exception from this function.
    """
    market = get_market(market_code)
    today = datetime.now(market.timezone).date()
    if not market.is_trading_day(today):
        logger.info("jobs.scheduler.holiday_skip", market=market_code, date=today.isoformat())
        return None

    try:
        report = await run_daily_pipeline(market_code)
    except Exception as exc:  # a bad run must never kill the scheduler
        logger.error("jobs.scheduler.pipeline_failed", market=market_code, error=str(exc))
        return None

    try:
        path = save_report(report)
        logger.info("jobs.scheduler.report_saved", market=market_code, path=str(path))
    except Exception as exc:  # report I/O failure shouldn't discard a good run
        logger.error("jobs.scheduler.save_report_failed", market=market_code, error=str(exc))

    # Paper-trading cycle: fill yesterday's orders against today's bars,
    # apply exit rules, queue new orders from today's picks, snapshot
    # equity. ``run_paper_cycle`` exception-contains each of its own steps,
    # so this outer guard is belt-and-braces against anything it missed
    # (e.g. failing to open the store) -- a paper-cycle failure must never
    # take down the scheduler or discard an otherwise-good screen run.
    settings = get_settings()
    store = BarStore(settings.duckdb_path)
    try:
        await run_paper_cycle(market_code, report, store)
    except Exception as exc:
        logger.error("jobs.scheduler.paper_cycle_failed", market=market_code, error=str(exc))
    finally:
        store.close()

    return report


async def _run_us_post_close() -> None:
    await run_market_job(US_NASDAQ.code)
    await run_market_job(US_NYSE.code)


async def _run_india_post_close() -> None:
    await run_market_job(IN_NSE.code)


def _parse_hour_minute(value: str) -> tuple[int, int]:
    hour_str, minute_str = value.split(":")
    return int(hour_str), int(minute_str)


def build_scheduler(settings: AppSettings | None = None) -> AsyncIOScheduler:
    """Build (but do not start) the scheduler with the two post-close jobs."""
    settings = settings or get_settings()
    scheduler = AsyncIOScheduler()

    us_hour, us_minute = _parse_hour_minute(settings.scheduler.us_post_close)
    scheduler.add_job(
        _run_us_post_close,
        trigger=CronTrigger(
            hour=us_hour,
            minute=us_minute,
            day_of_week="mon-fri",
            timezone=settings.scheduler.us_timezone,
        ),
        id="us_post_close",
        name="US post-close screen (NASDAQ + NYSE)",
    )

    india_hour, india_minute = _parse_hour_minute(settings.scheduler.india_post_close)
    scheduler.add_job(
        _run_india_post_close,
        trigger=CronTrigger(
            hour=india_hour,
            minute=india_minute,
            day_of_week="mon-fri",
            timezone=settings.scheduler.india_timezone,
        ),
        id="india_post_close",
        name="India post-close screen (NSE)",
    )

    return scheduler


def start_scheduler(settings: AppSettings | None = None) -> AsyncIOScheduler:
    """Build and (if enabled in settings) start the scheduler."""
    settings = settings or get_settings()
    scheduler = build_scheduler(settings)
    if settings.scheduler.enabled:
        scheduler.start()
    else:
        logger.info("jobs.scheduler.disabled")
    return scheduler
