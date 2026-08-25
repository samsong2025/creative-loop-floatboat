from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..services.brand_registry import BrandRegistry
from ..services.residual_qc import build_residual_report
from ..services.watermark_engine import analyze_video


class WatermarkAnalyzeRequest(BaseModel):
    source_relative_path: str
    brand_id: str
    detector_fps: float = Field(default=2.0, ge=0.1, le=10.0)
    icon_threshold: float = Field(default=0.72, ge=0.0, le=1.0)
    max_frames: int = Field(default=1000, ge=1, le=10000)
    enable_ocr_confirmation: bool = True
    max_reacquire_gap_seconds: float = Field(default=3.0, ge=0.5, le=10.0)
    enable_floatboat_uncertain: bool = False


class ResidualScanRequest(WatermarkAnalyzeRequest):
    residual_threshold: float = Field(default=0.80, ge=0.0, le=1.0)


class WatermarkRenderRequest(BaseModel):
    source_relative_path: str
    brand_id: str
    dynamic_tracks: list[dict] = Field(default_factory=list)
    output_dir_relative_path: str | None = None
    watermark_asset_relative_path: str | None = None
    own_brand_text: str = ""
    render_enabled: bool = False
    min_render_confidence: float = Field(default=0.80, ge=0.0, le=1.0)
    blur_sigma: float = Field(default=5.0, ge=0.8, le=32.0)


class WatermarkRepairRequest(ResidualScanRequest):
    output_dir_relative_path: str | None = None
    watermark_asset_relative_path: str | None = None
    own_brand_text: str = ""


def register_watermark_routes(app, workspace_dir: Path, registry: BrandRegistry) -> None:
    router = APIRouter(prefix="/watermark", tags=["watermark-v2"])

    def source_path(relative: str) -> Path:
        root = Path(workspace_dir).resolve()
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise HTTPException(status_code=404, detail="Source video not found inside workspace")
        return path

    def profile_or_404(brand_id: str) -> dict:
        profile = registry.get_profile(brand_id)
        if not profile:
            raise HTTPException(status_code=404, detail=f"Brand profile not found: {brand_id}")
        profile["_workspace_dir"] = str(workspace_dir)
        return profile

    def legacy_track_adapter(tracks: list[dict], frame_width: int, frame_height: int) -> list[dict]:
        """Convert canonical v1 segments into safe legacy renderer timelines.

        Each JUMP/reappearance segment becomes a separate legacy track.  This
        is intentional: the old renderer linearly interpolates waypoints and
        must never be given one path that crosses a screen-space jump.
        """
        adapted = []
        for track in tracks:
            raw_segments = track.get("segments") or [{"samples": track.get("path", []), "max_interpolation_gap_seconds": 1.25}]
            for segment_index, segment in enumerate(raw_segments, 1):
                path = []
                for node in segment.get("samples", []):
                    bbox = node.get("bbox", [])
                    if len(bbox) != 4:
                        continue
                    x, y, box_width, box_height = (float(value) for value in bbox)
                    # Engine candidates use pixels.  The legacy follow renderer
                    # uses normalized [x1,y1,x2,y2] boxes.
                    path.append(
                        {
                            "t": float(node.get("t", 0.0)),
                            "bbox": [
                                max(0.0, min(1.0, x / max(1, frame_width))),
                                max(0.0, min(1.0, y / max(1, frame_height))),
                                max(0.0, min(1.0, (x + box_width) / max(1, frame_width))),
                                max(0.0, min(1.0, (y + box_height) / max(1, frame_height))),
                            ],
                        }
                    )
                if path:
                    adapted.append(
                        {
                            "track_id": f"{track.get('track_id', track.get('brand_id', 'brand'))}-seg-{segment_index:02d}",
                            "waypoints": path,
                            "mean_identity_confidence": float(track.get("confidence", 0.8)),
                            "max_interpolation_gap_seconds": float(segment.get("max_interpolation_gap_seconds") or 1.25),
                            "motion": segment.get("motion", "continuous"),
                        }
                    )
        return adapted

    @router.post("/analyze")
    async def analyze(payload: WatermarkAnalyzeRequest):
        path = source_path(payload.source_relative_path)
        profile = profile_or_404(payload.brand_id)
        return await asyncio.to_thread(analyze_video, path, profile, detector_fps=payload.detector_fps, threshold=payload.icon_threshold, max_frames=payload.max_frames, enable_ocr_confirmation=payload.enable_ocr_confirmation, max_reacquire_gap_seconds=payload.max_reacquire_gap_seconds)

    @router.post("/residual-scan")
    async def residual_scan(payload: ResidualScanRequest):
        path = source_path(payload.source_relative_path)
        profile = profile_or_404(payload.brand_id)
        analysis = await asyncio.to_thread(analyze_video, path, profile, detector_fps=payload.detector_fps, threshold=payload.icon_threshold, max_frames=payload.max_frames, enable_ocr_confirmation=payload.enable_ocr_confirmation, max_reacquire_gap_seconds=payload.max_reacquire_gap_seconds)
        report = build_residual_report(analysis, threshold=payload.residual_threshold)
        return {"ok": True, "report": report, "analysis_metrics": analysis.get("metrics", {})}

    @router.post("/render")
    async def render(payload: WatermarkRenderRequest):
        # The historical implementation rejected render_enabled=true but still
        # rendered when it was false, then auto-filled own_brand_text from the
        # profile.  That inverted gate created repeated product/brand plates on
        # dynamic tracks.  This endpoint now has one behavior for either value:
        # render verified tracks as bounded feathered blur, never as branding.
        path = source_path(payload.source_relative_path)
        profile = profile_or_404(payload.brand_id)
        unsafe_tracks = [
            track.get("track_id", "unknown")
            for track in payload.dynamic_tracks
            if float(track.get("confidence", 0.0)) < payload.min_render_confidence
            or not (
                str(track.get("state", "")) in {"CONFIRMED", "FADING", "COASTING", "REACQUIRED"}
                or "CONFIRMED" in str(track.get("source", ""))
            )
        ]
        # Legacy callers may omit v1 state/segments but must still satisfy the
        # existing explicit CONFIRMED source contract.  New tracks have a
        # state plus canonical segment samples.
        if unsafe_tracks:
            raise HTTPException(
                status_code=422,
                detail={
                    "reason": "low_confidence_or_unconfirmed_tracks_cannot_render",
                    "minimum_confidence": payload.min_render_confidence,
                    "rejected_track_ids": unsafe_tracks[:20],
                    "next_action": "submit the candidate ROIs to /agent/watermark/decision before creating a render timeline",
                },
            )
        try:
            from ..branding_moving_watermark_insight import MovingWatermarkFollowRequest, _render_follow_cover
            from .. import branding_moving_watermark_insight as legacy_renderer
        except ImportError:
            from branding_moving_watermark_insight import MovingWatermarkFollowRequest, _render_follow_cover
            import branding_moving_watermark_insight as legacy_renderer
        output_dir = payload.output_dir_relative_path or f"processed/watermark_v2/{payload.brand_id}"
        root = Path(workspace_dir).resolve()
        target_dir = (root / output_dir).resolve()
        if root not in target_dir.parents and target_dir != root:
            raise HTTPException(status_code=422, detail="Output directory must be inside workspace")
        target_dir.mkdir(parents=True, exist_ok=True)
        output_path = target_dir / f"{path.stem}.watermark-v2.mp4"
        cap = legacy_renderer.cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise HTTPException(status_code=422, detail="Could not open source video")
        meta = {"width": int(cap.get(legacy_renderer.cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(legacy_renderer.cv2.CAP_PROP_FRAME_HEIGHT)), "fps": float(cap.get(legacy_renderer.cv2.CAP_PROP_FPS) or 30.0), "duration_seconds": float(cap.get(legacy_renderer.cv2.CAP_PROP_FRAME_COUNT) or 0) / max(1.0, float(cap.get(legacy_renderer.cv2.CAP_PROP_FPS) or 30.0))}
        cap.release()
        legacy_tracks = legacy_track_adapter(payload.dynamic_tracks, meta["width"], meta["height"])
        if not legacy_tracks:
            raise HTTPException(status_code=422, detail="At least one v1 DynamicWatermarkTrack segment with [x, y, width, height] samples is required")
        request = MovingWatermarkFollowRequest(
            relative_path=payload.source_relative_path,
            competitor_name=str(profile.get("product_name") or payload.brand_id),
            # Retain false in the forwarded legacy request: the value is now
            # informational only, and _render_follow_cover is blur-only.
            render_enabled=False,
            output_dir_relative_path=output_dir,
            blur_sigma=float(payload.blur_sigma),
        )
        report = await asyncio.to_thread(_render_follow_cover, path, output_path, legacy_tracks, None, request, meta)
        if report.get("return_code") != 0 or report.get("write_error"):
            raise HTTPException(status_code=500, detail={"render_failed": report})
        return {
            "ok": True,
            "brand_id": payload.brand_id,
            "source": "dynamic_watermark_engine_v1_segment_adapter_blur_only",
            "treatment": "feathered_gaussian_blur",
            "own_brand_overlay": False,
            "render": report,
        }

    @router.post("/repair")
    async def repair(payload: WatermarkRepairRequest):
        # The v2 local-repair endpoint intentionally returns exact failing ranges
        # and a re-analysis plan; a caller supplies or reviews track geometry
        # before deterministic rendering, preventing unbounded whole-video reruns.
        path = source_path(payload.source_relative_path)
        profile = profile_or_404(payload.brand_id)
        analysis = await asyncio.to_thread(analyze_video, path, profile, detector_fps=payload.detector_fps, threshold=payload.icon_threshold, max_frames=payload.max_frames, enable_ocr_confirmation=payload.enable_ocr_confirmation, max_reacquire_gap_seconds=payload.max_reacquire_gap_seconds)
        report = build_residual_report(analysis, threshold=payload.residual_threshold)
        return {"ok": True, "repair_required": not report["pass"], "brand_id": payload.brand_id, "intervals": report["intervals"], "next_action": "POST /watermark/render with dynamic_tracks limited to these intervals" if not report["pass"] else "no_repair_needed", "report": report}

    app.include_router(router)
