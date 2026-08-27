"""``argus.data.sources``: market-scoped universe resolution and provider
cleanup around ``check_source_health``.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

import argus.data.sources as sources_module
from argus.config import AppSettings
from argus.data.prices.base import BAR_DTYPE, PriceDataProvider, ProviderHealth, Quote
from argus.data.sources import check_source_health, create_source, resolve_universe_provider
from argus.data.universe import SeedUniverseProvider, TVUniverseProvider
from argus.db import init_db
from argus.db.models import DataSource
from argus.markets import IN_NSE, US_NASDAQ, Instrument, Market


def _settings(tmp_path: Path) -> AppSettings:
    return AppSettings(data_dir=tmp_path, _env_file=None)  # type: ignore[call-arg]


# --- resolve_universe_provider: markets_json scoping -------------------------


async def test_resolve_universe_provider_respects_source_market_scoping(
    tmp_path: Path,
) -> None:
    """A tvscreener source scoped to IN_NSE only must not be used for US
    resolution -- US falls back to seeds, IN gets the TV-backed provider."""
    settings = _settings(tmp_path)
    await init_db(settings)
    await create_source(
        name="tv-india-only",
        kind="tvscreener",
        markets=[IN_NSE.code],
        config={},
        settings=settings,
    )

    us_provider = await resolve_universe_provider(US_NASDAQ.code, settings)
    in_provider = await resolve_universe_provider(IN_NSE.code, settings)

    assert isinstance(us_provider, SeedUniverseProvider)
    assert isinstance(in_provider, TVUniverseProvider)


async def test_resolve_universe_provider_empty_markets_list_covers_every_market(
    tmp_path: Path,
) -> None:
    """An empty ``markets`` list means "all markets" -- the tvscreener source
    should be used regardless of the requested market code."""
    settings = _settings(tmp_path)
    await init_db(settings)
    await create_source(
        name="tv-all-markets",
        kind="tvscreener",
        markets=[],
        config={},
        settings=settings,
    )

    us_provider = await resolve_universe_provider(US_NASDAQ.code, settings)
    in_provider = await resolve_universe_provider(IN_NSE.code, settings)

    assert isinstance(us_provider, TVUniverseProvider)
    assert isinstance(in_provider, TVUniverseProvider)


async def test_resolve_universe_provider_no_tv_source_returns_seeds(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    await init_db(settings)

    provider = await resolve_universe_provider(US_NASDAQ.code, settings)

    assert isinstance(provider, SeedUniverseProvider)


# --- check_source_health: closes a throwaway provider -------------------------


class _CloseableFakeProvider:
    """Minimal ``PriceDataProvider`` implementation that tracks whether it
    was closed, mirroring ``NSEProvider``'s ``httpx.AsyncClient`` ownership."""

    name = "fake"

    def __init__(self, *, raise_on_health_check: bool = False) -> None:
        self.closed = False
        self._raise_on_health_check = raise_on_health_check

    def supports(self, market: Market) -> bool:
        return True

    async def get_daily_bars(
        self, inst: Instrument, start: date, end: date
    ) -> NDArray[np.void]:
        return np.zeros(0, dtype=BAR_DTYPE)

    async def get_quote(self, inst: Instrument) -> Quote | None:
        return None

    async def health_check(self) -> ProviderHealth:
        if self._raise_on_health_check:
            raise RuntimeError("simulated failure")
        return ProviderHealth(ok=True, detail="ok", checked_at=datetime.now(UTC))

    async def aclose(self) -> None:
        self.closed = True


def _patch_build_price_provider(
    monkeypatch: pytest.MonkeyPatch, fake: PriceDataProvider
) -> None:
    monkeypatch.setattr(sources_module, "build_price_provider", lambda kind, config: fake)


async def test_check_source_health_closes_throwaway_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    await init_db(settings)
    source = await create_source(
        name="nse-1", kind="nse", markets=[IN_NSE.code], config={}, settings=settings
    )

    fake = _CloseableFakeProvider()
    _patch_build_price_provider(monkeypatch, fake)

    health = await check_source_health(source, settings)

    assert health.ok is True
    assert fake.closed is True


async def test_check_source_health_closes_provider_even_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The throwaway provider must be closed even if ``health_check`` raises."""
    settings = _settings(tmp_path)
    await init_db(settings)
    source = await create_source(
        name="nse-2", kind="nse", markets=[IN_NSE.code], config={}, settings=settings
    )

    fake = _CloseableFakeProvider(raise_on_health_check=True)
    _patch_build_price_provider(monkeypatch, fake)

    with pytest.raises(RuntimeError, match="simulated failure"):
        await check_source_health(source, settings)

    assert fake.closed is True


async def test_data_source_markets_json_shape_sanity(tmp_path: Path) -> None:
    """Sanity check that ``markets_json`` round-trips as expected -- guards
    the ``_source_covers_market`` helper's assumption about its shape."""
    settings = _settings(tmp_path)
    await init_db(settings)
    source = await create_source(
        name="check", kind="tvscreener", markets=[IN_NSE.code], config={}, settings=settings
    )
    assert isinstance(source, DataSource)
    assert source.markets_json == {"markets": [IN_NSE.code]}
