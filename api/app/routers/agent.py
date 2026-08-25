from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..services.brand_registry import BrandRegistry


class WatermarkAIDecisionRequest(BaseModel):
    """Schema-validated callback for uncertain local watermark candidates only."""

    brand_present: bool
    brand_id: str
    confidence: float = Field(ge=0.0, le=1.0)
    recommended_bbox: list[float] | None = Field(default=None, min_length=4, max_length=4)
    candidate_id: str | None = None


def register_agent_watermark_routes(app, registry: BrandRegistry) -> None:
    router = APIRouter(prefix="/agent/watermark", tags=["agent-watermark"])

    @router.post("/decision")
    async def watermark_decision(payload: WatermarkAIDecisionRequest):
        if not registry.get_profile(payload.brand_id):
            raise HTTPException(status_code=404, detail=f"Unknown brand_id: {payload.brand_id}")
        if payload.brand_present and payload.recommended_bbox is None:
            raise HTTPException(status_code=422, detail="recommended_bbox is required when brand_present is true")
        return {
            "ok": True,
            "accepted": True,
            "decision": payload.model_dump(),
            "routing": "caller_may_merge_validated_decision_into_watermark_timeline",
        }

    app.include_router(router)