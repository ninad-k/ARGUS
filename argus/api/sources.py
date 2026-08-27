"""Data-source CRUD endpoints -- the "multiple sources" admin API.

Thin wrappers over ``argus.data.sources`` (which owns the actual DB reads/
writes and the health-check call) -- the NiceGUI sources page uses the same
functions directly, so the CRUD logic itself lives in exactly one place.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from argus.api.schemas import DataSourceCreate, DataSourceOut, DataSourceUpdate, SourceTestResult
from argus.config import get_settings
from argus.data.sources import (
    check_source_health,
    create_source,
    delete_source,
    get_source,
    list_sources,
    update_source,
)
from argus.db.models import DataSource

router = APIRouter(prefix="/api/v1/sources", tags=["sources"])


def _source_out(source: DataSource) -> DataSourceOut:
    return DataSourceOut(
        id=source.id,
        name=source.name,
        kind=source.kind,
        markets=list(source.markets_json.get("markets", [])),
        config=dict(source.config_json),
        priority=source.priority,
        enabled=source.enabled,
        last_health=source.last_health,
        created_at=source.created_at,
    )


@router.get("", response_model=list[DataSourceOut])
async def list_all_sources() -> list[DataSourceOut]:
    settings = get_settings()
    sources = await list_sources(settings)
    return [_source_out(s) for s in sources]


@router.post("", response_model=DataSourceOut, status_code=201)
async def create_new_source(body: DataSourceCreate) -> DataSourceOut:
    settings = get_settings()
    source = await create_source(
        name=body.name,
        kind=body.kind,
        markets=body.markets,
        config=body.config,
        priority=body.priority,
        settings=settings,
    )
    return _source_out(source)


@router.patch("/{source_id}", response_model=DataSourceOut)
async def patch_source(source_id: int, body: DataSourceUpdate) -> DataSourceOut:
    settings = get_settings()
    source = await update_source(
        source_id,
        name=body.name,
        kind=body.kind,
        markets=body.markets,
        config=body.config,
        priority=body.priority,
        enabled=body.enabled,
        settings=settings,
    )
    if source is None:
        raise HTTPException(status_code=404, detail=f"source {source_id} not found")
    return _source_out(source)


@router.delete("/{source_id}", status_code=204)
async def delete_existing_source(source_id: int) -> None:
    settings = get_settings()
    deleted = await delete_source(source_id, settings)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"source {source_id} not found")


@router.post("/{source_id}/test", response_model=SourceTestResult)
async def test_source(source_id: int) -> SourceTestResult:
    settings = get_settings()
    source = await get_source(source_id, settings)
    if source is None:
        raise HTTPException(status_code=404, detail=f"source {source_id} not found")
    health = await check_source_health(source, settings)
    return SourceTestResult(ok=health.ok, detail=health.detail, checked_at=health.checked_at)
