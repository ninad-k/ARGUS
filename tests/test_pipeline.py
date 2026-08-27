"""End-to-end ``run_daily_pipeline`` against injected Static providers.

Settings are built directly (no env file, ``llm.enabled=False`` by default)
and ``argus.pipeline.get_settings`` is monkeypatched to return them, so these
tests write to ``tmp_path`` and never touch the network -- matching the
"no network in default tests" project rule.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from sqlalchemy import select

from argus.config import AppSettings
from argus.config.data import DataSettings
from argus.config.llm import LLMSettings
from argus.data.prices.base import BAR_DTYPE, ProviderHealth, Quote
from argus.data.prices.static_provider import StaticPriceProvider, synthetic_bars
from argus.data.universe import StaticUniverseProvider
from argus.db import async_session
from argus.db.models import ScreenRun
from argus.markets import US_NASDAQ, Instrument, Market
from argus.pipeline import run_daily_pipeline

_TODAY = date.today()  # noqa: DTZ011 -- matches refresh_bars' own daily-cache boundary


def _settings(tmp_path: Path, *, llm_enabled: bool = False) -> AppSettings:
    return AppSettings(
        data_dir=tmp_path,
        llm=LLMSettings(enabled=llm_enabled, _env_file=None),  # type: ignore[call-arg]
        _env_file=None,  # type: ignore[call-arg]
    )


def _provider_and_instruments(
    market_code: str = US_NASDAQ.code,
) -> tuple[StaticPriceProvider, list[Instrument]]:
    lookback = 299
    start = _TODAY - timedelta(days=lookback)
    bars_by_symbol = {
        "MOMO": synthetic_bars(n=300, start_price=100.0, seed=1, start=start, trend=0.006),
        "FLAT": synthetic_bars(n=300, start_price=80.0, seed=2, start=start, trend=0.0),
    }
    provider = StaticPriceProvider(bars_by_symbol)
    instruments = [Instrument(symbol=s, market_code=market_code) for s in bars_by_symbol]
    return provider, instruments


class _FlakyProvider:
    """Wraps a ``StaticPriceProvider`` but raises for one chosen symbol."""

    name = "flaky"

    def __init__(self, inner: StaticPriceProvider, bad_symbol: str) -> None:
        self._inner = inner
        self._bad_symbol = bad_symbol

    def supports(self, market: Market) -> bool:
        return True

    async def get_daily_bars(self, inst: Instrument, start: date, end: date) -> NDArray[np.void]:
        if inst.symbol == self._bad_symbol:
            raise RuntimeError("simulated provider failure")
        return await self._inner.get_daily_bars(inst, start, end)

    async def get_quote(self, inst: Instrument) -> Quote | None:
        return await self._inner.get_quote(inst)

    async def health_check(self) -> ProviderHealth:
        return await self._inner.health_check()


class _HangingProvider:
    """A provider whose ``get_daily_bars`` never returns.

    Used to exercise the belt-and-braces outer ``asyncio.wait_for`` in
    ``_refresh_all``/``_one`` -- a provider that ignores its own
    timeout/retry budget entirely must still not be able to hang the whole
    pipeline (see ``argus.pipeline._per_symbol_refresh_timeout``).
    """

    name = "hanging"

    def supports(self, market: Market) -> bool:
        return True

    async def get_daily_bars(self, inst: Instrument, start: date, end: date) -> NDArray[np.void]:
        await asyncio.Event().wait()  # never set -- hangs forever
        return np.zeros(0, dtype=BAR_DTYPE)  # pragma: no cover -- unreachable

    async def get_quote(self, inst: Instrument) -> Quote | None:
        return None  # pragma: no cover -- not exercised by the pipeline path used here

    async def health_check(self) -> ProviderHealth:
        return ProviderHealth(ok=True, detail="ok", checked_at=datetime.now(UTC))


async def test_run_daily_pipeline_full_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    monkeypatch.setattr("argus.pipeline.get_settings", lambda: settings)

    provider, instruments = _provider_and_instruments()
    universe_provider = StaticUniverseProvider({US_NASDAQ.code: instruments})

    report = await run_daily_pipeline(
        US_NASDAQ.code,
        provider=provider,
        universe_provider=universe_provider,
    )

    assert report.run_id > 0
    assert report.bars_refreshed > 0
    assert report.symbols_failed == []
    assert report.llm_used is False
    assert report.result.market_code == US_NASDAQ.code
    assert any(c.instrument.symbol == "MOMO" for c in report.result.top)

    async with async_session(settings) as session:
        run = (
            await session.execute(select(ScreenRun).where(ScreenRun.id == report.run_id))
        ).scalar_one()
        assert run.market == US_NASDAQ.code
        assert run.status == "completed"


async def test_run_daily_pipeline_second_run_adds_no_new_bars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    monkeypatch.setattr("argus.pipeline.get_settings", lambda: settings)

    provider, instruments = _provider_and_instruments()
    universe_provider = StaticUniverseProvider({US_NASDAQ.code: instruments})

    first = await run_daily_pipeline(
        US_NASDAQ.code, provider=provider, universe_provider=universe_provider
    )
    assert first.bars_refreshed > 0

    second = await run_daily_pipeline(
        US_NASDAQ.code, provider=provider, universe_provider=universe_provider
    )
    assert second.bars_refreshed == 0
    assert second.run_id > first.run_id
    assert any(c.instrument.symbol == "MOMO" for c in second.result.top)


async def test_run_daily_pipeline_llm_false_sets_llm_used_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # settings.llm.enabled=True here -- only the explicit `llm=False` argument
    # should stop the pipeline from attempting any LLM review.
    settings = _settings(tmp_path, llm_enabled=True)
    monkeypatch.setattr("argus.pipeline.get_settings", lambda: settings)

    provider, instruments = _provider_and_instruments()
    universe_provider = StaticUniverseProvider({US_NASDAQ.code: instruments})

    report = await run_daily_pipeline(
        US_NASDAQ.code,
        provider=provider,
        universe_provider=universe_provider,
        llm=False,
    )

    assert report.llm_used is False
    assert all(c.llm_verdict is None for c in report.result.candidates)


async def test_run_daily_pipeline_llm_review_applied_when_backend_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A working (fake, in-process) backend should get wired through and applied."""
    settings = _settings(tmp_path, llm_enabled=True)
    monkeypatch.setattr("argus.pipeline.get_settings", lambda: settings)

    from argus.advisor.llm import LLMRequest, LLMResponse

    class _FakeBackend:
        provider = "fake"
        model = "fake-1"

        async def complete(self, req: LLMRequest) -> LLMResponse:
            return LLMResponse(
                text='{"picks": [{"symbol": "MOMO", "verdict": "buy", '
                '"confidence": 90, "thesis": "t", "risks": "r"}]}',
                provider=self.provider,
                model=self.model,
            )

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr("argus.pipeline.build_backend", lambda _settings: _FakeBackend())

    provider, instruments = _provider_and_instruments()
    universe_provider = StaticUniverseProvider({US_NASDAQ.code: instruments})

    report = await run_daily_pipeline(
        US_NASDAQ.code, provider=provider, universe_provider=universe_provider
    )

    assert report.llm_used is True
    momo = next(c for c in report.result.candidates if c.instrument.symbol == "MOMO")
    assert momo.llm_verdict is not None
    assert momo.llm_verdict.verdict == "buy"


async def test_run_daily_pipeline_council_review_applied_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``council_enabled=True`` fans the review out to the default council
    (buffett, lynch, druckenmiller) instead of the single-pass reviewer --
    same fake backend answers every persona's call identically here, so
    fusion collapses to a unanimous "buy" with all three votes attached."""
    settings = AppSettings(
        data_dir=tmp_path,
        llm=LLMSettings(enabled=True, council_enabled=True, _env_file=None),  # type: ignore[call-arg]
        _env_file=None,  # type: ignore[call-arg]
    )
    monkeypatch.setattr("argus.pipeline.get_settings", lambda: settings)

    from argus.advisor.llm import LLMRequest, LLMResponse

    class _FakeCouncilBackend:
        provider = "fake"
        model = "fake-1"

        async def complete(self, req: LLMRequest) -> LLMResponse:
            return LLMResponse(
                text='{"picks": [{"symbol": "MOMO", "verdict": "buy", '
                '"confidence": 80, "thesis": "t"}]}',
                provider=self.provider,
                model=self.model,
            )

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr("argus.pipeline.build_backend", lambda _settings: _FakeCouncilBackend())

    provider, instruments = _provider_and_instruments()
    universe_provider = StaticUniverseProvider({US_NASDAQ.code: instruments})

    report = await run_daily_pipeline(
        US_NASDAQ.code, provider=provider, universe_provider=universe_provider
    )

    assert report.llm_used is True
    momo = next(c for c in report.result.candidates if c.instrument.symbol == "MOMO")
    assert momo.llm_verdict is not None
    assert momo.llm_verdict.verdict == "buy"
    assert len(momo.llm_verdict.votes) == 3  # default council: buffett, lynch, druckenmiller


async def test_run_daily_pipeline_llm_backend_failure_degrades_gracefully(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path, llm_enabled=True)
    monkeypatch.setattr("argus.pipeline.get_settings", lambda: settings)

    def _raise(_settings: Any) -> Any:
        raise RuntimeError("backend construction failed")

    monkeypatch.setattr("argus.pipeline.build_backend", _raise)

    provider, instruments = _provider_and_instruments()
    universe_provider = StaticUniverseProvider({US_NASDAQ.code: instruments})

    report = await run_daily_pipeline(
        US_NASDAQ.code, provider=provider, universe_provider=universe_provider
    )

    assert report.llm_used is False
    assert report.run_id > 0


async def test_run_daily_pipeline_symbol_failure_does_not_abort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    monkeypatch.setattr("argus.pipeline.get_settings", lambda: settings)

    base_provider, instruments = _provider_and_instruments()
    bad_instrument = Instrument(symbol="BAD", market_code=US_NASDAQ.code)
    instruments.append(bad_instrument)
    flaky = _FlakyProvider(base_provider, bad_symbol="BAD")
    universe_provider = StaticUniverseProvider({US_NASDAQ.code: instruments})

    report = await run_daily_pipeline(
        US_NASDAQ.code, provider=flaky, universe_provider=universe_provider
    )

    assert report.symbols_failed == ["BAD"]
    assert report.run_id > 0
    assert any(c.instrument.symbol == "MOMO" for c in report.result.top)


async def test_run_daily_pipeline_outer_timeout_guards_hung_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A provider that never returns must not hang the whole pipeline -- the
    belt-and-braces outer timeout in ``_refresh_all`` should catch it and
    record the symbol as failed instead of blocking ``asyncio.gather``
    forever."""
    # Zero out the fixed slack so the computed per-symbol timeout is tiny and
    # this test runs fast, rather than waiting out the real 30s default.
    monkeypatch.setattr("argus.pipeline._PER_SYMBOL_TIMEOUT_SLACK_SECONDS", 0.0)

    settings = AppSettings(
        data_dir=tmp_path,
        data=DataSettings(provider_timeout_seconds=0.05, provider_max_retries=0),
        llm=LLMSettings(enabled=False, _env_file=None),  # type: ignore[call-arg]
        _env_file=None,  # type: ignore[call-arg]
    )
    monkeypatch.setattr("argus.pipeline.get_settings", lambda: settings)

    instruments = [Instrument(symbol="HANG", market_code=US_NASDAQ.code)]
    universe_provider = StaticUniverseProvider({US_NASDAQ.code: instruments})

    report = await run_daily_pipeline(
        US_NASDAQ.code,
        provider=_HangingProvider(),
        universe_provider=universe_provider,
    )

    assert report.symbols_failed == ["HANG"]
    assert report.bars_refreshed == 0
    assert report.run_id > 0
