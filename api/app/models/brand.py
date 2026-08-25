from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class BrandIdentity(BaseModel):
    """Explicit product identity captured from one Insight detail acquisition."""

    brand_id: str | None = None
    product_name: str | None = None
    app_title: str | None = None
    package_name: str | None = None
    aliases: list[str] = Field(default_factory=list)
    icon_url: str | None = None
    icon_source: str | None = None
    detail_id: str | None = None
    identity_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    identity_source: str = "unresolved"


class BrandProfile(BaseModel):
    """Persisted brand asset profile used by the watermark engine."""

    brand_id: str
    product_name: str
    app_title: str | None = None
    package_names: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    icon: dict[str, Any] = Field(default_factory=dict)
    templates: dict[str, Any] = Field(default_factory=dict)
    detector_strategy: list[str] = Field(
        default_factory=lambda: [
            "icon_grayscale_match",
            "edge_template_match",
            "multi_scale_match",
            "roi_ocr_confirmation",
        ]
    )
    registry_version: int = 1
    updated_at: str


class WatermarkCandidate(BaseModel):
    """A single low-frequency visual/OCR observation at one sampled time.

    ``bbox`` deliberately remains in pixel space.  Detector output is easier to
    inspect in pixels, while the renderer adapter is responsible for converting
    it to the normalized coordinate system used by the legacy compositor.
    """

    time_seconds: float
    brand_id: str
    bbox: list[float] = Field(min_length=4, max_length=4)
    icon_similarity: float = Field(default=0.0, ge=0.0, le=1.0)
    edge_similarity: float = Field(default=0.0, ge=0.0, le=1.0)
    scale_factor: float = Field(default=1.0, ge=0.0)
    detector_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    ocr_brand_match: float = Field(default=0.0, ge=0.0, le=1.0)
    temporal_consistency: float = Field(default=0.0, ge=0.0, le=1.0)
    spatial_prior: float = Field(default=0.0, ge=0.0, le=1.0)
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    scene_id: int = 0
    observed: bool = True
    source: str = "icon_match"


class WatermarkTrack(BaseModel):
    """Renderer-facing dynamic watermark track.

    ``segments`` is the v1 canonical representation.  ``path`` is retained as
    a backwards-compatible diagnostic/adapter field for existing callers.
    Segment boundaries are hard boundaries: a renderer must never interpolate
    from one segment to the next, especially for a JUMP or re-acquisition.
    """

    track_id: str
    brand_id: str
    kind: str = "dynamic_watermark"
    start_seconds: float
    end_seconds: float
    confidence: float = Field(ge=0.0, le=1.0)
    source: str = "icon_match+tracker"
    state: str = "TENTATIVE"
    scene_id: int = 0
    motion_summary: list[str] = Field(default_factory=list)
    anchor_count: int = 0
    segments: list[dict[str, Any]] = Field(default_factory=list)
    path: list[dict[str, Any]] = Field(default_factory=list)


# Public v1 name.  Keep the old model name as the implementation type so
# existing integrations importing WatermarkTrack continue to work unchanged.
DynamicWatermarkTrack = WatermarkTrack


class ResidualReport(BaseModel):
    brand_id: str
    pass_: bool = Field(alias="pass")
    max_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    intervals: list[dict[str, Any]] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}