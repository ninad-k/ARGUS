"""``suggest_for_picks`` fan-out over a ``ScreenReport``'s top picks (using an
injected provider factory so nothing here touches the network) and
``persist_suggestions``' DB round-trip."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from sqlalchemy import select

from argus.config import AppSettings
from argus.config.options import OptionsSettings
from argus.data.prices.base import ProviderHealth
from argus.db import async_session, init_db
from argus.db.models import DailyPick, OptionSuggestion, ScreenRun
from argus.markets import US_NASDAQ, Instrument
from argus.options.models import OptionChain, OptionQuote
from argus.options.providers.base import OptionChainProvider, StaticOptionChainProvider
from argus.options.suggester import RiskLevel, persist_suggestions, suggest_for_picks
from argus.pipeline import ScreenReport
from argus.screener.base import Candidate
from argus.screener.runner import ScreenResult

_EXPIRY = date.today() + timedelta(days=45)  # noqa: DTZ011 -- matches suggester's own convention
_OPTS = OptionsSettings(_env_file=None)


def _settings(tmp_path: Path) -> AppSettings:
    return AppSettings(data_dir=tmp_path, _env_file=None)


def _candidate(symbol: str, *, has_options: bool = True, has_futures: bool = False) -> Candidate:
    inst = Instrument(
        symbol=symbol,
        market_code=US_NASDAQ.code,
        has_options=has_options,
        has_futures=has_futures,
    )
    return Candidate(instrument=inst, strategy="momentum", score=80.0, direction="long")


def _quote(strike: float, *, delta: float = 0.60) -> OptionQuote:
    return OptionQuote(
        strike=strike,
        expiry=_EXPIRY,
        right="C",
        bid=9.5,
        ask=10.5,
        last=10.0,
        iv=0.30,
        oi=500.0,
        delta=delta,
    )


def _chain(symbol: str) -> OptionChain:
    return OptionChain(
        symbol=symbol,
        market_code=US_NASDAQ.code,
        spot=120.0,
        as_of=datetime.now(UTC),
        expiries=[_EXPIRY],
        quotes=[_quote(110.0)],
    )


class _NoneChainProvider:
    """Serves no chain for any instrument -- used to prove a symbol whose
    provider returns ``None`` is simply absent from the result, not an
    error."""

    name = "none"

    def supports(self, inst: Instrument) -> bool:
        return True

    async def list_expiries(self, inst: Instrument) -> list[date]:
        return []

    async def get_chain(self, inst: Instrument, expiry: date | None = None) -> OptionChain | None:
        return None

    async def health_check(self) -> ProviderHealth:
        return ProviderHealth(ok=True, detail="ok", checked_at=datetime.now(UTC))

    async def aclose(self) -> None:
        return None


class _ClosingSpyStaticProvider:
    """Wraps a ``StaticOptionChainProvider`` and records whether ``aclose``
    was called -- proves ``suggest_for_picks`` always closes the provider it
    built, success or failure."""

    name = "closing_spy"

    def __init__(self, inner: StaticOptionChainProvider) -> None:
        self._inner = inner
        self.closed = False

    def supports(self, inst: Instrument) -> bool:
        return self._inner.supports(inst)

    async def list_expiries(self, inst: Instrument) -> list[date]:
        return await self._inner.list_expiries(inst)

    async def get_chain(self, inst: Instrument, expiry: date | None = None) -> OptionChain | None:
        return await self._inner.get_chain(inst, expiry)

    async def health_check(self) -> ProviderHealth:
        return await self._inner.health_check()

    async def aclose(self) -> None:
        self.closed = True


def _report(candidates: list[Candidate], *, run_id: int = 1) -> ScreenReport:
    result = ScreenResult(
        market_code=US_NASDAQ.code,
        run_ts=datetime.now(UTC),
        universe_size=10,
        filtered_size=len(candidates),
        candidates=candidates,
        top=candidates,
    )
    return ScreenReport(
        result=result, run_id=run_id, bars_refreshed=0, symbols_failed=[], llm_used=False
    )


async def test_suggest_for_picks_returns_suggestions_for_option_flagged_picks() -> None:
    candidates = [_candidate("AAPL"), _candidate("MSFT")]
    report = _report(candidates)

    providers = {
        "AAPL": StaticOptionChainProvider({("AAPL", US_NASDAQ.code): _chain("AAPL")}),
        "MSFT": StaticOptionChainProvider({("MSFT", US_NASDAQ.code): _chain("MSFT")}),
    }

    def factory(inst: Instrument) -> OptionChainProvider | None:
        return providers.get(inst.symbol)

    suggestions = await suggest_for_picks(
        report, risk=RiskLevel.CONSERVATIVE, settings=_OPTS, provider_factory=factory
    )

    assert set(suggestions) == {"AAPL", "MSFT"}
    assert suggestions["AAPL"].instrument_type == "CE"
    assert suggestions["AAPL"].strike == 110.0


async def test_suggest_for_picks_skips_pick_without_option_flags() -> None:
    candidates = [_candidate("AAPL"), _candidate("PLAIN", has_options=False, has_futures=False)]
    report = _report(candidates)

    called: list[str] = []

    def factory(inst: Instrument) -> OptionChainProvider | None:
        called.append(inst.symbol)
        return StaticOptionChainProvider({(inst.symbol, US_NASDAQ.code): _chain(inst.symbol)})

    suggestions = await suggest_for_picks(
        report, risk=RiskLevel.CONSERVATIVE, settings=_OPTS, provider_factory=factory
    )

    assert "PLAIN" not in called
    assert "PLAIN" not in suggestions
    assert "AAPL" in suggestions


async def test_suggest_for_picks_skips_symbol_whose_provider_returns_no_chain() -> None:
    candidates = [_candidate("AAPL"), _candidate("NOCHAIN")]
    report = _report(candidates)

    providers: dict[str, OptionChainProvider] = {
        "AAPL": StaticOptionChainProvider({("AAPL", US_NASDAQ.code): _chain("AAPL")}),
        "NOCHAIN": _NoneChainProvider(),
    }

    def factory(inst: Instrument) -> OptionChainProvider | None:
        return providers.get(inst.symbol)

    suggestions = await suggest_for_picks(
        report, risk=RiskLevel.CONSERVATIVE, settings=_OPTS, provider_factory=factory
    )

    assert "NOCHAIN" not in suggestions
    assert "AAPL" in suggestions


async def test_suggest_for_picks_closes_providers_it_built() -> None:
    candidates = [_candidate("AAPL")]
    report = _report(candidates)

    spy = _ClosingSpyStaticProvider(
        StaticOptionChainProvider({("AAPL", US_NASDAQ.code): _chain("AAPL")})
    )

    def factory(inst: Instrument) -> OptionChainProvider | None:
        return spy

    await suggest_for_picks(
        report, risk=RiskLevel.CONSERVATIVE, settings=_OPTS, provider_factory=factory
    )

    assert spy.closed is True


async def test_suggest_for_picks_no_provider_for_instrument_is_skipped() -> None:
    candidates = [_candidate("UNSUPPORTED")]
    report = _report(candidates)

    def factory(inst: Instrument) -> OptionChainProvider | None:
        return None

    suggestions = await suggest_for_picks(
        report, risk=RiskLevel.CONSERVATIVE, settings=_OPTS, provider_factory=factory
    )

    assert suggestions == {}


async def test_persist_suggestions_round_trips_rows_linked_to_correct_picks(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    await init_db(settings)

    async with async_session(settings) as session:
        run = ScreenRun(
            market=US_NASDAQ.code,
            run_ts=datetime.now(UTC),
            universe_size=2,
            strategies_json={"strategies": ["momentum"]},
            status="completed",
            duration_ms=1,
        )
        session.add(run)
        await session.flush()

        aapl_pick = DailyPick(
            run_id=run.id,
            symbol="AAPL",
            market=US_NASDAQ.code,
            strategy="momentum",
            score=90.0,
            stage="breakout",
            features_json={},
            created_at=run.run_ts,
        )
        msft_pick = DailyPick(
            run_id=run.id,
            symbol="MSFT",
            market=US_NASDAQ.code,
            strategy="momentum",
            score=85.0,
            stage="breakout",
            features_json={},
            created_at=run.run_ts,
        )
        session.add_all([aapl_pick, msft_pick])
        await session.commit()
        run_id = run.id
        aapl_pick_id = aapl_pick.id
        msft_pick_id = msft_pick.id

    candidates = [_candidate("AAPL"), _candidate("MSFT")]
    report = _report(candidates, run_id=run_id)
    providers = {
        "AAPL": StaticOptionChainProvider({("AAPL", US_NASDAQ.code): _chain("AAPL")}),
        "MSFT": StaticOptionChainProvider({("MSFT", US_NASDAQ.code): _chain("MSFT")}),
    }

    def factory(inst: Instrument) -> OptionChainProvider | None:
        return providers.get(inst.symbol)

    suggestions = await suggest_for_picks(
        report, risk=RiskLevel.CONSERVATIVE, settings=_OPTS, provider_factory=factory
    )
    assert len(suggestions) == 2

    inserted = await persist_suggestions(run_id, suggestions, RiskLevel.CONSERVATIVE, settings)
    assert inserted == 2

    async with async_session(settings) as session:
        rows = (
            (await session.execute(select(OptionSuggestion))).scalars().all()
        )
        by_pick_id = {r.pick_id: r for r in rows}
        assert len(rows) == 2
        assert by_pick_id[aapl_pick_id].instrument_type == "CE"
        assert by_pick_id[aapl_pick_id].strike == 110.0
        assert by_pick_id[aapl_pick_id].risk_level == "conservative"
        assert by_pick_id[msft_pick_id].instrument_type == "CE"


async def test_persist_suggestions_skips_symbol_with_no_matching_pick(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    await init_db(settings)

    async with async_session(settings) as session:
        run = ScreenRun(
            market=US_NASDAQ.code,
            run_ts=datetime.now(UTC),
            universe_size=1,
            strategies_json={"strategies": ["momentum"]},
            status="completed",
            duration_ms=1,
        )
        session.add(run)
        await session.flush()
        run_id = run.id
        await session.commit()

    candidates = [_candidate("GHOST")]
    report = _report(candidates, run_id=run_id)
    providers = {"GHOST": StaticOptionChainProvider({("GHOST", US_NASDAQ.code): _chain("GHOST")})}

    def factory(inst: Instrument) -> OptionChainProvider | None:
        return providers.get(inst.symbol)

    suggestions = await suggest_for_picks(
        report, risk=RiskLevel.CONSERVATIVE, settings=_OPTS, provider_factory=factory
    )
    assert "GHOST" in suggestions

    inserted = await persist_suggestions(run_id, suggestions, RiskLevel.CONSERVATIVE, settings)
    assert inserted == 0


async def test_persist_suggestions_empty_dict_is_a_no_op(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    await init_db(settings)
    inserted = await persist_suggestions(1, {}, RiskLevel.MODERATE, settings)
    assert inserted == 0
