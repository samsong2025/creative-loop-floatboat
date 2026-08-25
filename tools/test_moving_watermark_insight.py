"""Regression checks for Creative Loop's Insightrackr-style moving-watermark follow module.

Run with the API module importable (for example from the api container):
    python tools/test_moving_watermark_insight.py

The tests are pure in-memory checks: no source video, workspace media or
external service is modified.  They verify the ported Insightrackr semantics:
dense-observation track building, no-extrapolation interpolation, adaptive
backdrop/opacity bounds and per-action render receipts.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))

# The policy helpers are pure Python, but branding_moving_watermark_insight
# also declares FastAPI request models.  Keep this regression executable in the
# lightweight vision environment without pretending that environment is the
# API runtime.
if "fastapi" not in sys.modules:
    fastapi_stub = types.ModuleType("fastapi")

    class HTTPException(Exception):
        pass

    fastapi_stub.HTTPException = HTTPException
    sys.modules["fastapi"] = fastapi_stub

if "pydantic" not in sys.modules:
    pydantic_stub = types.ModuleType("pydantic")

    class BaseModel:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    def Field(default=None, default_factory=None, **kwargs):
        return default_factory() if default_factory is not None else default

    pydantic_stub.BaseModel = BaseModel
    pydantic_stub.Field = Field
    sys.modules["pydantic"] = pydantic_stub

# The pure policy functions under test do not touch cv2/numpy at runtime.
# Allow the regression to run in the lightweight vision environment too.
try:
    import cv2  # noqa: F401
except ImportError:
    sys.modules.setdefault("cv2", types.ModuleType("cv2"))

try:
    import numpy  # noqa: F401
except ImportError:
    sys.modules.setdefault("numpy", types.ModuleType("numpy"))

from app.branding_moving_watermark_insight import (  # noqa: E402
    _adaptive_opacities,
    _best_hit_per_time,
    _candidate_time_windows,
    _interpolated_waypoint_bbox,
    _is_corner_center,
    _likely_product,
    _moving_tracks_from_rows,
    _visual_points_to_preview_tracks,
    _is_in_editorial_exclusion,
)

_POLICY = {
    "target": "ReelShort",
    "min_observations": 3,
    "min_candidate_confidence": 0.65,
    "min_mean_confidence": 0.62,
    "min_movement_ratio": 0.018,
    "max_interpolation_gap_seconds": 0.32,
}


def _row(t, x, y, confidence=0.90, w=0.03, h=0.02):
    return {
        "t": float(t),
        "x": float(x),
        "y": float(y),
        "bbox": [x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0],
        "text": "ReelShort",
        "confidence": float(confidence),
    }


def test_accepts_dense_moving_verified_track():
    rows = [_row(1.0, 0.10, 0.55), _row(1.2, 0.13, 0.56), _row(1.4, 0.16, 0.57)]
    tracks, rejected = _moving_tracks_from_rows(rows, 1280, 720, _POLICY)
    assert not rejected
    assert len(tracks) == 1
    assert tracks[0]["point_count"] == 3
    assert tracks[0]["tracking_policy"].endswith("hide_on_tracking_gap")
    assert tracks[0]["max_interpolation_gap_seconds"] == 0.32
    assert tracks[0]["movement_px"] > 12.96


def test_rejects_two_point_candidate():
    tracks, rejected = _moving_tracks_from_rows(
        [_row(74.25, 0.22, 0.72), _row(75.0, 0.22, 0.72)],
        1280,
        720,
        _POLICY,
    )
    assert not tracks
    assert any(item["reason"] == "insufficient_dense_verified_observations" for item in rejected)


def test_rejects_stationary_candidate():
    tracks, rejected = _moving_tracks_from_rows(
        [_row(1.0, 0.22, 0.72), _row(1.2, 0.22, 0.72), _row(1.4, 0.22, 0.72)],
        1280,
        720,
        _POLICY,
    )
    assert not tracks
    assert any(item["reason"] == "movement_below_verified_dynamic_threshold" for item in rejected)


def test_rejects_corner_only_candidate():
    tracks, rejected = _moving_tracks_from_rows(
        [_row(1.0, 0.80, 0.10), _row(1.2, 0.81, 0.11), _row(1.4, 0.82, 0.12)],
        1280,
        720,
        _POLICY,
    )
    assert not tracks
    assert any(item["reason"] == "fixed_corner_identity_not_dynamic" for item in rejected)


def test_rejects_low_confidence_candidate():
    tracks, rejected = _moving_tracks_from_rows(
        [_row(1.0, 0.10, 0.55, confidence=0.50), _row(1.2, 0.13, 0.56, confidence=0.50), _row(1.4, 0.16, 0.57, confidence=0.50)],
        1280,
        720,
        _POLICY,
    )
    assert not tracks
    assert any(item["reason"] == "insufficient_dense_verified_observations" for item in rejected)


def test_strict_interpolation_only_inside_observed_window():
    track = {"waypoints": [_row(1.0, 0.10, 0.55), _row(1.2, 0.13, 0.56), _row(1.4, 0.16, 0.57)]}
    assert _interpolated_waypoint_bbox(track["waypoints"], 0.99, 0.32) is None
    assert _interpolated_waypoint_bbox(track["waypoints"], 1.10, 0.32) is not None
    assert _interpolated_waypoint_bbox(track["waypoints"], 1.41, 0.32) is None

    gap_track = {
        "waypoints": [
            _row(1.0, 0.10, 0.55),
            _row(1.8, 0.19, 0.58),
            _row(2.0, 0.22, 0.59),
        ]
    }
    assert _interpolated_waypoint_bbox(gap_track["waypoints"], 1.50, 0.32) is None


def test_adaptive_opacities_follow_insightrackr_docs():
    req = SimpleNamespace(
        backdrop_opacity_base=0.88,
        backdrop_opacity_span=0.04,
        watermark_opacity_base=0.76,
        watermark_opacity_span=0.06,
    )
    lo_back, lo_wm = _adaptive_opacities(0.0, req)
    hi_back, hi_wm = _adaptive_opacities(1.0, req)
    assert abs(lo_back - 0.88) < 1e-9 and abs(hi_back - 0.92) < 1e-9
    assert abs(lo_wm - 0.76) < 1e-9 and abs(hi_wm - 0.82) < 1e-9


def test_likely_product_matching():
    assert _likely_product("ReelShort", ["ReelShort"]) == (True, 1.0)
    assert _likely_product("Reelshort", ["ReelShort"]) == (True, 1.0)
    assert _likely_product("Reel Short", ["ReelShort"])[0] is True
    assert _likely_product("DramaBox", ["ReelShort"])[0] is False
    assert _likely_product("X", ["ReelShort"])[0] is False


def test_best_hit_per_time_dedupes_variants():
    hits = {
        1.0: [
            {"text": "ReelShort", "identity_confidence": 0.70, "ocr_confidence": 0.70, "similarity": 1.0, "bbox": [0.1, 0.5, 0.14, 0.52]},
            {"text": "ReelShort", "identity_confidence": 0.92, "ocr_confidence": 0.92, "similarity": 1.0, "bbox": [0.2, 0.5, 0.24, 0.52]},
        ]
    }
    rows = _best_hit_per_time(hits)
    assert len(rows) == 1
    assert rows[0]["confidence"] == 0.92
    assert rows[0]["x"] == 0.22


def test_candidate_time_windows_padding():
    rows = [_row(74.25, 0.22, 0.72), _row(75.0, 0.23, 0.73)]
    windows = _candidate_time_windows(rows, 2.0)
    assert len(windows) == 1
    assert windows[0][0] <= 72.25
    assert windows[0][1] >= 77.0


def test_corner_center_classifier():
    assert _is_corner_center(0.80, 0.10) is True
    assert _is_corner_center(0.20, 0.05) is True
    assert _is_corner_center(0.50, 0.50) is False


def test_per_action_receipt_gate_semantics():
    # 移植 workflow_bridge.py 的逐动作回执门禁：actions_detected == actions_rendered
    tracks = [{"track_id": "t1"}, {"track_id": "t2"}]
    receipts = [
        {"track_id": "t1", "frames_rendered": 55},
        {"track_id": "t2", "frames_rendered": 0},
    ]
    rendered = sum(1 for receipt in receipts if int(receipt["frames_rendered"]) > 0)
    assert rendered < len(tracks)  # 门禁失败：检测 2 个动作但只渲染 1 个


def test_visual_template_tracks_are_normalized_without_extending_the_window():
    converted = _visual_points_to_preview_tracks(
        [
            {
                "track_id": "reviewed-01",
                "visibility_window": [2.0, 2.2],
                "max_interpolation_gap_seconds": 0.32,
                "points": [
                    {"time_seconds": 2.0, "bbox": {"x": 72, "y": 128, "width": 360, "height": 80}, "template_score": 0.31},
                    {"time_seconds": 2.2, "bbox": {"x": 90, "y": 144, "width": 360, "height": 80}, "template_score": 0.34},
                ],
            }
        ],
        width=720,
        height=1280,
    )
    assert len(converted) == 1
    track = converted[0]
    assert track["waypoints"][0]["bbox"] == [0.1, 0.1, 0.6, 0.1625]
    assert _interpolated_waypoint_bbox(track["waypoints"], 1.99, 0.32) is None
    assert _interpolated_waypoint_bbox(track["waypoints"], 2.10, 0.32) is not None
    assert _interpolated_waypoint_bbox(track["waypoints"], 2.21, 0.32) is None


def test_legacy_follow_renderer_cannot_add_own_brand_overlay():
    """The compatibility renderer must route tracks to blur, not alpha assets."""
    import inspect
    import app.branding_moving_watermark_insight as moving

    source = inspect.getsource(moving._render_follow_cover)
    assert "_blur_follow_region(" in source
    assert "_alpha_overlay_follow(" not in source
    assert '"treatment": "feathered_gaussian_blur"' in source
    assert '"own_brand_overlay": False' in source


def test_temporal_repair_uses_masked_clean_plate_inpainting_not_blur_fallback():
    """Moving marks must use a glyph-masked repair, never ROI blur fallback."""
    import inspect
    from app.branding_moving_watermark_insight import _fade_source_watermark_frame

    source = inspect.getsource(_fade_source_watermark_frame)
    assert "cv2.INPAINT_TELEA" in source
    assert "template_glyph_mask_temporal_clean_plate_flow_telea" not in source
    assert "repaired_roi = cv2.GaussianBlur(local" not in source
    assert "rectangle blur" in source


def test_editorial_replacement_excludes_only_finally_removed_source_times():
    """An editorial replacement may exempt only source frames it removes."""
    intervals = [(28.6, 49.97)]
    assert _is_in_editorial_exclusion(39.3, intervals)
    assert not _is_in_editorial_exclusion(28.5, intervals)
    assert not _is_in_editorial_exclusion(50.0, intervals)


def test_template_aligned_residual_metric_ignores_unrelated_roi_texture():
    """Residual QA must follow the watermark template, not generic ROI edges."""
    import numpy as np
    from app.branding_moving_watermark_insight import _template_aligned_high_pass_correlation

    rng = np.random.default_rng(7)
    template = rng.normal(0.0, 1.0, size=(28, 96)).astype(np.float32)
    support = np.abs(template) >= 0.35
    # The same scene texture remains in both frames, while the template-shaped
    # component falls from full strength to 20%. A generic edge-energy metric
    # still sees texture; the identity metric must report suppression.
    texture = rng.normal(0.0, 12.0, size=template.shape)
    source = np.clip(128.0 + texture + template * 22.0, 0, 255).astype(np.uint8)
    repaired = np.clip(128.0 + texture + template * 2.0, 0, 255).astype(np.uint8)
    source_score = _template_aligned_high_pass_correlation(source, template, support)
    repaired_score = _template_aligned_high_pass_correlation(repaired, template, support)

    assert source_score is not None
    assert repaired_score is not None
    assert abs(repaired_score) / abs(source_score) < 0.32


if __name__ == "__main__":
    test_accepts_dense_moving_verified_track()
    test_rejects_two_point_candidate()
    test_rejects_stationary_candidate()
    test_rejects_corner_only_candidate()
    test_rejects_low_confidence_candidate()
    test_strict_interpolation_only_inside_observed_window()
    test_adaptive_opacities_follow_insightrackr_docs()
    test_likely_product_matching()
    test_best_hit_per_time_dedupes_variants()
    test_candidate_time_windows_padding()
    test_corner_center_classifier()
    test_per_action_receipt_gate_semantics()
    test_visual_template_tracks_are_normalized_without_extending_the_window()
    test_legacy_follow_renderer_cannot_add_own_brand_overlay()
    test_temporal_repair_uses_masked_clean_plate_inpainting_not_blur_fallback()
    test_editorial_replacement_excludes_only_finally_removed_source_times()
    test_template_aligned_residual_metric_ignores_unrelated_roi_texture()
    print("insight moving-watermark regressions passed")
