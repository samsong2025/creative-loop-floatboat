from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..services.brand_identity import identity_from_detail
from ..services.brand_registry import BrandRegistry


class BrandResolveRequest(BaseModel):
    product_name: str | None = None
    app_title: str | None = None
    package_name: str | None = None


class BrandRegisterRequest(BrandResolveRequest):
    aliases: list[str] = Field(default_factory=list)
    icon_url: str | None = None
    icon_source: str | None = None
    detail_id: str | None = None


class BrandRefreshAssetsRequest(BaseModel):
    icon_url: str | None = None


def register_brand_routes(app, registry: BrandRegistry) -> None:
    router = APIRouter(prefix="/brands", tags=["brands"])

    @router.get("")
    async def list_brands():
        return {"ok": True, "brands": registry.list_profiles(), "registry": "config/brand_registry"}

    @router.get("/{brand_id}")
    async def get_brand(brand_id: str):
        profile = registry.get_profile(brand_id)
        if not profile:
            raise HTTPException(status_code=404, detail=f"Brand not found: {brand_id}")
        return {"ok": True, "profile": profile}

    @router.post("/resolve")
    async def resolve_brand(payload: BrandResolveRequest):
        profile = registry.resolve(product_name=payload.product_name, app_title=payload.app_title, package_name=payload.package_name)
        return {"ok": True, "resolved": bool(profile), "profile": profile}

    @router.post("/register")
    async def register_brand(payload: BrandRegisterRequest):
        identity = identity_from_detail(payload.model_dump())
        if not identity.product_name:
            raise HTTPException(status_code=422, detail="Explicit product_name or app_title is required")
        profile = registry.upsert(identity)
        return {"ok": True, "profile": profile}

    @router.post("/{brand_id}/refresh-assets")
    async def refresh_assets(brand_id: str, payload: BrandRefreshAssetsRequest):
        profile = registry.get_profile(brand_id)
        if not profile:
            raise HTTPException(status_code=404, detail=f"Brand not found: {brand_id}")
        icon_url = payload.icon_url or profile.get("icon", {}).get("source_url")
        if not icon_url:
            raise HTTPException(status_code=422, detail="No explicit icon URL available for this brand")
        try:
            import httpx
            response = await asyncio.to_thread(httpx.get, icon_url, timeout=20.0, follow_redirects=True)
            response.raise_for_status()
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Icon refresh failed: {type(exc).__name__}") from exc
        identity = identity_from_detail({"product_name": profile["product_name"], "app_title": profile.get("app_title"), "package_name": (profile.get("package_names") or [None])[0], "icon_url": icon_url, "icon_source": "insight_detail_refresh"})
        return {"ok": True, "profile": registry.upsert(identity, icon_bytes=response.content, source_url=icon_url)}

    app.include_router(router)