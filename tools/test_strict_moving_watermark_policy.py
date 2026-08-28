"""Regression checks for Creative Loop's strict moving-watermark policy.

Run with the API module importable (for example from the api container):
    python tools/test_strict_moving_watermark_policy.py

The tests are pure in-memory checks: no source video, workspace media or
external service is modified.
"""
from __future__ import annotations

import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))

# The policy helpers are pure Python, but branding_v09 also declares FastAPI
# request models.  Keep this regression executable in the lightweight vision
# environment without pretending that environment is the API runtime.
if "fastapi" not in sys.modules or not hasattr(sys.modules.get("fastapi"), "Request"):
    fastapi_stub = types.ModuleType("fastapi")

    class HTTPException(Exception):
        pass

    class Request:
        pass

    fastapi_stub.HTTPException = HTTPException
    fastapi_stub.Request = Request
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

from app.branding_v09 import (
    _diag_merge_word_fragments,
    _diag_signature_similarity,
    _dynamic_visual_template_paths,
    _load_diagonal_brand_cover_geometry,
    _operator_apply_dynamic_brand_cover,
    _operator_apply_dynamic_wordmark_blur,
    _operator_dynamic_wordmark_stroke_mask,
    _operator_dynamic_brand_tracks_from_census,
    _operator_interpolated_dynamic_bbox,
    _source_icon_watermark_hit,
    _strict_verified_identity_tracks,
    _strict_verified_visual_tracks,
)


def _request(**overrides):
    defaults = {
        "dynamic_verified_identity_mode": True,
        "dynamic_verified_identity_min_observations": 3,
        "dynamic_verified_identity_min_mean_confidence": 0.62,
        "dynamic_verified_identity_min_movement_ratio": 0.018,
        "dynamic_verified_identity_max_interpolation_gap_seconds": 0.32,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _hit(t, x, y, confidence=0.90):
    return {
        "time_seconds": t,
        "target": "ReelShort",
        "observed_text": "ReelShort",
        "similarity": 1.0,
        "exact": True,
        "ocr_confidence": confidence,
        "bbox": {"x": x, "y": y, "width": 120, "height": 28},
    }


def test_accepts_dense_moving_verified_identity_track():
    tracks, policy, rejected = _strict_verified_identity_tracks(
        [_hit(1.0, 80, 400), _hit(1.2, 105, 415), _hit(1.4, 132, 430)],
        720,
        1280,
        _request(),
    )
    # The effective gap is raised to the sampling cadence when necessary;
    # never require a narrower interval than the detector can observe.
    assert 0.32 <= policy["max_interpolation_gap_seconds"] <= 2.0
    assert not rejected
    assert len(tracks) == 1
    assert tracks[0]["movement_px"] > 12.96
    assert tracks[0]["tracking_policy"].endswith("hide_on_tracking_gap")


def test_rejects_two_point_or_stationary_candidate():
    tracks, _, rejected = _strict_verified_identity_tracks(
        [_hit(74.25, 284, 928), _hit(75.0, 284, 928)],
        720,
        1280,
        _request(),
    )
    assert not tracks
    assert rejected
    assert rejected[0]["reason"] == "insufficient_dense_verified_observations"


def test_rejects_terminal_card_track():
    tracks, _, rejected = _strict_verified_identity_tracks(
        [
            {**_hit(97.5, 113, 647), "bbox": {"x": 113, "y": 647, "width": 495, "height": 91}},
            {**_hit(97.8, 145, 655), "bbox": {"x": 145, "y": 655, "width": 450, "height": 82}},
            {**_hit(98.1, 170, 662), "bbox": {"x": 170, "y": 662, "width": 333, "height": 45}},
        ],
        720,
        1280,
        _request(),
        duration=98.667,
    )
    assert not tracks
    assert any(item["reason"] == "terminal_end_card_not_dynamic_watermark" for item in rejected)


def test_census_strict_track_is_handed_to_renderer(tmp_path=None):
    """The direct-render handoff must consume strict tracks, not just plan totals."""
    import json
    import app.branding_v09 as branding

    root = Path(tempfile.mkdtemp()) if tmp_path is None else Path(tmp_path)
    old_workspace = branding.WORKSPACE
    try:
        branding.WORKSPACE = root
        report = {
            "video": {"width": 720, "height": 1280},
            "scan": {"effective_interval_seconds": 0.30},
            "verified_dynamic_identity": {
                "enabled": True,
                "tracks": [
                    {
                        "track_id": "strict-ocr-reelshort-01",
                        "classification": "verified_moving_competitor_watermark",
                        "handler": "verified_identity_dense_ocr_track",
                        "target": "ReelShort",
                        "max_interpolation_gap_seconds": 0.32,
                        "tracking_policy": "verified_identity_only__no_pre_or_post_extrapolation__hide_on_tracking_gap",
                        "points": [_hit(1.0, 80, 400), _hit(1.2, 105, 415), _hit(1.4, 132, 430)],
                    }
                ],
            },
        }
        path = root / "census.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        tracks = _operator_dynamic_brand_tracks_from_census("census.json")
        assert len(tracks) == 1
        assert tracks[0]["cluster_id"] == "strict-ocr-reelshort-01"
        assert tracks[0]["max_interpolation_gap_seconds"] == 0.32
        assert _operator_interpolated_dynamic_bbox(tracks[0], 1.10, 720, 1280, 2.5)
        assert _operator_interpolated_dynamic_bbox(tracks[0], 0.99, 720, 1280, 2.5) is None
    finally:
        branding.WORKSPACE = old_workspace


def test_same_source_diagonal_profile_returns_renderable_centers():
    """Validated grid profiles must use the center schema consumed by renderer."""
    import json
    import app.branding_v09 as branding

    root = Path(tempfile.mkdtemp())
    old_workspace = branding.WORKSPACE
    try:
        branding.WORKSPACE = root
        (root / "router.json").write_text(
            json.dumps(
                {
                    "source": {"relative_path": "raw/clip.mp4"},
                    "watermark_layers": [],
                }
            ),
            encoding="utf-8",
        )
        (root / "config").mkdir()
        (root / "config" / "watermark_profiles.json").write_text(
            json.dumps(
                {
                    "profiles": [
                        {
                            "profile_id": "clip-grid",
                            "handler": "diagonal_tiled_grid",
                            "source": {"relative_path": "raw/clip.mp4"},
                            "geometry": {
                                "origin_normalized": {"x": 0.3, "y": 0.8},
                                "basis_vectors_normalized": [
                                    {"dx": -0.14, "dy": -0.21},
                                    {"dx": 0.34, "dy": -0.05},
                                ],
                                "canonical_tile_normalized": {"width": 0.27, "height": 0.05},
                            },
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        req = SimpleNamespace(
            diagonal_brand_cover_enabled=True,
            diagonal_brand_cover_router_report_relative_path="router.json",
            diagonal_brand_cover_max_points=24,
        )
        geometry = _load_diagonal_brand_cover_geometry(req, 720, 1280)
        assert geometry["available"] is True
        assert geometry["points"] and "center" in geometry["points"][0]
        assert geometry["geometry_source"] == "validated_same_source_profile"
    finally:
        branding.WORKSPACE = old_workspace


def test_strict_renderer_hides_outside_visibility_and_gaps():
    track = {
        "tracking_policy": "verified_identity_only__no_pre_or_post_extrapolation__hide_on_tracking_gap",
        "max_interpolation_gap_seconds": 0.32,
        "points": [
            {"time_seconds": 1.0, "bbox": {"x": 80, "y": 400, "width": 120, "height": 28}},
            {"time_seconds": 1.2, "bbox": {"x": 105, "y": 415, "width": 120, "height": 28}},
            {"time_seconds": 1.4, "bbox": {"x": 132, "y": 430, "width": 120, "height": 28}},
        ],
    }
    assert _operator_interpolated_dynamic_bbox(track, 0.99, 720, 1280, 2.0) is None
    assert _operator_interpolated_dynamic_bbox(track, 1.10, 720, 1280, 2.0)
    assert _operator_interpolated_dynamic_bbox(track, 1.41, 720, 1280, 2.0) is None

    gap_track = {
        **track,
        "points": [
            track["points"][0],
            {"time_seconds": 1.8, "bbox": {"x": 160, "y": 445, "width": 120, "height": 28}},
            {"time_seconds": 2.0, "bbox": {"x": 184, "y": 460, "width": 120, "height": 28}},
        ],
    }
    assert _operator_interpolated_dynamic_bbox(gap_track, 1.40, 720, 1280, 2.0) is None


def test_visual_detector_keeps_reviewed_wordmark_template_alongside_source_icon(tmp_path=None):
    """An app icon must not suppress the reviewed moving-wordmark template."""
    import app.branding_v09 as branding

    root = Path(tempfile.mkdtemp()) if tmp_path is None else Path(tmp_path)
    old_workspace = branding.WORKSPACE
    try:
        branding.WORKSPACE = root
        (root / "config").mkdir(parents=True)
        fallback = root / "config" / "dynamic_watermark_reference_reelshort_v1.png"
        fallback.write_bytes(b"reviewed-wordmark")
        icon = root / "config" / "source-icon.png"
        icon.write_bytes(b"source-icon")
        req = SimpleNamespace(source_icon_relative_path="config/source-icon.png", source_logo_relative_path=None)
        paths = _dynamic_visual_template_paths(req)
        assert icon in paths
        assert fallback in paths
    finally:
        branding.WORKSPACE = old_workspace


def test_dynamic_cover_is_blur_only_without_own_brand_pixels():
    """Dynamic wordmarks are feather-blurred without text, logo or a plate."""
    import numpy as np
    import app.branding_v09 as branding

    frame = np.full((140, 240, 3), 96, dtype=np.uint8)
    # Synthetic high-contrast letter strokes within a tracked wordmark box.
    frame[48:92, 80:84] = 240
    frame[48:52, 80:128] = 240
    frame[68:72, 80:122] = 240
    frame[48:92, 118:122] = 240
    bbox = {"x": 74, "y": 42, "width": 60, "height": 56}
    track = {
        "cluster_id": "dynamic-test",
        "tracking_policy": "verified_identity_only__no_pre_or_post_extrapolation__hide_on_tracking_gap",
        "max_interpolation_gap_seconds": 0.32,
        "points": [
            {"time_seconds": 1.0, "bbox": bbox},
            {"time_seconds": 1.2, "bbox": bbox},
            {"time_seconds": 1.4, "bbox": bbox},
        ],
    }
    req = SimpleNamespace(
        dynamic_brand_cover_max_gap_seconds=0.32,
        dynamic_brand_cover_expand_ratio=1.0,
        dynamic_watermark_blur_sigma=5.0,
    )
    profile = {"brand_name": "Floatboat", "product_name": "Floatboat", "assets": {}}
    rendered, applied = _operator_apply_dynamic_brand_cover(frame, 1.2, [track], profile, 240, 140, req)
    assert len(applied) == 1
    assert applied[0]["treatment"] == "feathered_gaussian_blur"
    assert applied[0]["own_brand_overlay"] is False
    # The detected watermark strokes are softened, while pixels outside the
    # track remain untouched and no replacement text can be added.
    assert not np.array_equal(rendered[48:92, 80:128], frame[48:92, 80:128])
    assert np.array_equal(rendered[0:20, 0:20], frame[0:20, 0:20])


def test_dynamic_wordmark_blur_uses_strokes_not_whole_tracking_box():
    """A broad track may cross a face; only watermark-like strokes may blur."""
    import numpy as np

    frame = np.full((160, 280, 3), 96, dtype=np.uint8)
    # A wordmark-like group is confined to the upper-left of an intentionally
    # broad track box. The unrelated lower-right detail represents story pixels
    # (for example an actor's face) and must remain unchanged.
    frame[30:34, 38:130] = 240
    frame[34:60, 38:42] = 240
    frame[34:60, 126:130] = 240
    # Textured detail represents unrelated story pixels (for example a face).
    detail = np.indices((36, 44)).sum(axis=0) % 2
    frame[92:128, 176:220] = np.where(detail[..., None] == 0, 80, 180)
    # Match the reviewed reference aspect ratio so its glyph support stays in
    # the upper band of this intentionally broad tracking region.
    bbox = {"x": 24, "y": 20, "width": 220, "height": 44}
    rendered, detail = _operator_apply_dynamic_wordmark_blur(frame, bbox, 5.0)
    assert detail["mask_used"] is True
    assert detail["mask_source"] == "reviewed_template"
    assert not np.array_equal(rendered[30:60, 38:130], frame[30:60, 38:130])
    assert np.array_equal(rendered[92:128, 176:220], frame[92:128, 176:220])


def test_persistent_source_layout_can_be_disabled_for_dynamic_only_render():
    """A test render may opt out of broad fixed anchors without mutating config."""
    import json
    import app.branding_v09 as branding

    root = Path(tempfile.mkdtemp())
    old_workspace = branding.WORKSPACE
    try:
        branding.WORKSPACE = root
        (root / "router.json").write_text(json.dumps({"watermark_layers": []}), encoding="utf-8")
        (root / "config").mkdir()
        (root / "config" / "persistent_watermark_source_layouts.json").write_text(
            json.dumps({
                "sources": {
                    "raw/clip.mp4": {
                        "anchors": [{"id": "broad-fixed", "x": 0.1, "y": 0.1, "width": 0.5, "height": 0.3}]
                    }
                }
            }),
            encoding="utf-8",
        )
        plan = {"source": {"relative_path": "raw/clip.mp4"}}
        # The guard lives in the render handoff; validate that its source is
        # explicit so future refactors cannot silently restore the anchors.
        import inspect
        source = inspect.getsource(branding._replacement_render_sync)
        assert "persistent_watermark_auto_from_source_layout" in source
    finally:
        branding.WORKSPACE = old_workspace


def test_overlapping_dynamic_tracks_receive_shared_coverage_receipts():
    """One physical cover may satisfy multiple verified tracks on a frame."""
    import numpy as np

    frame = np.full((140, 240, 3), 96, dtype=np.uint8)
    bbox = {"x": 74, "y": 42, "width": 60, "height": 56}
    tracks = [
        {
            "cluster_id": "dynamic-primary",
            "_operator_dynamic_receipt_key": "dynamic-primary",
            "tracking_policy": "verified_identity_only__no_pre_or_post_extrapolation__hide_on_tracking_gap",
            "max_interpolation_gap_seconds": 0.32,
            "points": [
                {"time_seconds": 1.0, "bbox": bbox},
                {"time_seconds": 1.2, "bbox": bbox},
                {"time_seconds": 1.4, "bbox": bbox},
            ],
        },
        {
            "cluster_id": "dynamic-duplicate",
            "_operator_dynamic_receipt_key": "dynamic-duplicate",
            "tracking_policy": "verified_identity_only__no_pre_or_post_extrapolation__hide_on_tracking_gap",
            "max_interpolation_gap_seconds": 0.32,
            "points": [
                {"time_seconds": 1.0, "bbox": bbox},
                {"time_seconds": 1.2, "bbox": bbox},
                {"time_seconds": 1.4, "bbox": bbox},
            ],
        },
    ]
    req = SimpleNamespace(
        dynamic_brand_cover_max_gap_seconds=0.32,
        dynamic_brand_cover_expand_ratio=1.0,
    )
    profile = {"brand_name": "Floatboat", "product_name": "Floatboat", "assets": {}}
    _, applied = _operator_apply_dynamic_brand_cover(frame, 1.2, tracks, profile, 240, 140, req)
    assert [item["receipt_key"] for item in applied] == [
        "dynamic-primary",
        "dynamic-duplicate",
    ]
    assert applied[1]["treatment"] == "satisfied_by_shared_dynamic_blur"
    assert applied[1]["shared_blur_receipt_key"] == "dynamic-primary"


def test_dynamic_receipt_ledger_keeps_all_resolved_tracks_despite_bottom_actions():
    """Bottom treatment must not silently remove a verified track's receipt."""
    import ast
    import inspect
    import app.branding_v09 as branding

    source = inspect.getsource(branding._render_visual_base_full)
    tree = ast.parse(source)
    receipt_loop = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "track"
        and isinstance(node.iter, ast.Name)
        and node.iter.id == "dynamic_tracks"
    )
    bottom_filter_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_operator_filter_dynamic_tracks_covered_by_bottom_actions"
    ]
    assert receipt_loop.lineno > 0
    assert not bottom_filter_calls


def test_verified_dynamic_tracks_require_renderer_handoff():
    """A direct render must reject verified tracks before they can go untreated."""
    import ast
    import inspect
    import app.branding_v09 as branding

    source = inspect.getsource(branding._replacement_render_sync)
    tree = ast.parse(source)
    failures = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "HTTPException"
    ]
    rendered_source = "\n".join(
        ast.get_source_segment(source, node) or "" for node in failures
    )
    assert "dynamic_watermark_renderer_not_enabled" in rendered_source
    assert "expected_dynamic_receipt_tracks = expected_dynamic_tracks" in source


def test_fixed_layers_rebrand_while_dynamic_tracks_never_overlay():
    """The layer split is structural: fixed UI rebrands, moving marks repair."""
    import inspect
    import app.branding_v09 as branding

    full_render = inspect.getsource(branding._render_visual_base_full)
    bottom_start = full_render.index("bottom_applied = 0")
    diagonal_start = full_render.index("if (\n                bool(", bottom_start)
    bottom_branch = full_render[bottom_start:diagonal_start]
    assert "_draw_brand_banner(" in bottom_branch
    assert "_operator_apply_feathered_local_blur(" not in bottom_branch
    assert '"bottom_brand_treatment": "opaque_own_brand_banner"' in full_render

    persistent_render = inspect.getsource(branding._apply_diagonal_brand_cover)
    assert "_draw_brand_banner(" in persistent_render
    assert "_operator_apply_feathered_local_blur(" not in persistent_render

    dynamic_render = inspect.getsource(branding._operator_apply_dynamic_brand_cover)
    assert "_operator_apply_dynamic_wordmark_blur(" in dynamic_render
    assert "_draw_brand_banner(" not in dynamic_render
    assert '"own_brand_overlay": False' in dynamic_render


def test_dynamic_clean_repair_rejects_unverified_fade_fallback():
    """A residual-failing dynamic repair cannot become an output source."""
    import inspect
    import app.branding_moving_watermark_insight as moving
    import app.branding_v09 as branding

    temporal_source = inspect.getsource(moving.dynamic_watermark_temporal_repair_sync)
    assert "accepted_output = (fade_report.get(\"fade\") or {}) if qa_pass else None" in temporal_source
    assert '"fallback_policy": "reject_unverified_fade_preview"' in temporal_source
    assert '"status": "pass" if qa_pass else "review_required_before_phase_2"' in temporal_source

    render_source = inspect.getsource(branding._replacement_render_sync)
    assert "dynamic_watermark_clean_repair_not_verified" in render_source
    assert "dynamic_watermark_clean_repair_required" in render_source
    assert "blur or an own-brand overlay" in render_source


def test_temporal_repair_is_skipped_when_census_has_no_verified_tracks():
    """A clean source must not fail residual QA merely because it has no track."""
    import inspect
    import app.branding_v09 as branding

    render_source = inspect.getsource(branding._replacement_render_sync)
    assert "preflight_dynamic_tracks > 0" in render_source
    assert '"status": "skipped_no_verified_dynamic_tracks"' in render_source
    assert "not_applicable_no_verified_dynamic_tracks" in render_source


def test_production_execution_route_orders_watermark_before_editorial_actions():
    """Every production combination must take exactly one explicit route."""
    from app.branding_v09 import _production_execution_route

    def action(kind, **extra):
        return {"type": kind, "status": "AUTO", **extra}

    cases = [
        {
            "name": "clean_source_appends_own_end_card",
            "actions": [],
            "tracks": 0,
            "watermark": "none",
            "fixed": "skip_no_fixed_watermark",
            "dynamic": "skip_no_verified_dynamic_watermark",
            "mid": "skip",
            "end": "append_own",
        },
        {
            "name": "clean_source_replaces_midpromo_and_end_card",
            "actions": [action("mid_promo_replace"), action("end_card_replace")],
            "tracks": 0,
            "watermark": "none",
            "fixed": "skip_no_fixed_watermark",
            "dynamic": "skip_no_verified_dynamic_watermark",
            "mid": "replace",
            "end": "replace",
        },
        {
            "name": "fixed_only_cleans_then_rebrands",
            "actions": [action("fixed_brand_overlay")],
            "tracks": 0,
            "watermark": "fixed_only",
            "fixed": "source_cleanup_then_own_brand",
            "dynamic": "skip_no_verified_dynamic_watermark",
            "mid": "skip",
            "end": "append_own",
        },
        {
            "name": "dynamic_only_uses_temporal_repair",
            "actions": [action("moving_brand_overlay")],
            "tracks": 2,
            "watermark": "dynamic_only",
            "fixed": "skip_no_fixed_watermark",
            "dynamic": "temporal_clean_repair_no_overlay",
            "mid": "skip",
            "end": "append_own",
        },
        {
            "name": "fixed_and_dynamic_use_separate_treatments",
            "actions": [action("fixed_brand_overlay"), action("moving_brand_overlay")],
            "tracks": 1,
            "watermark": "fixed_and_dynamic",
            "fixed": "source_cleanup_then_own_brand",
            "dynamic": "temporal_clean_repair_no_overlay",
            "mid": "skip",
            "end": "append_own",
        },
        {
            "name": "end_card_without_midpromo_is_replaced",
            "actions": [action("end_card_replace")],
            "tracks": 0,
            "watermark": "none",
            "fixed": "skip_no_fixed_watermark",
            "dynamic": "skip_no_verified_dynamic_watermark",
            "mid": "skip",
            "end": "replace",
        },
    ]

    for case in cases:
        route = _production_execution_route(
            {"actions": case["actions"]},
            verified_dynamic_track_count=case["tracks"],
        )
        assert route["watermark"]["classification"] == case["watermark"], case["name"]
        assert route["watermark"]["fixed_strategy"] == case["fixed"], case["name"]
        assert route["watermark"]["dynamic_strategy"] == case["dynamic"], case["name"]
        assert route["editorial"]["mid_promo_strategy"] == case["mid"], case["name"]
        assert route["editorial"]["end_card_strategy"] == case["end"], case["name"]
        assert route["render"]["append_own_end_card_when_absent"] == (case["end"] == "append_own"), case["name"]

    dynamic_only = _production_execution_route(
        {"actions": [action("moving_brand_overlay")]},
        verified_dynamic_track_count=1,
    )
    assert dynamic_only["render"]["compose_top_brand_layers"] is False
    assert dynamic_only["render"]["diagonal_brand_cover_enabled"] is False
    assert dynamic_only["render"]["exclude_legacy_moving_actions"] is True

    unclassified = _production_execution_route(
        {"actions": []},
        detected_watermark_present=True,
    )
    assert unclassified["watermark"]["classification"] == "unclassified"
    assert unclassified["watermark"]["classification_required"] is True
    assert unclassified["watermark"]["fixed_strategy"] == "classification_required"


def test_production_route_does_not_execute_unconfirmed_review_actions():
    """Low-confidence semantic suggestions must not cut production footage."""
    from app.branding_v09 import _production_execution_route

    plan = {
        "actions": [
            {"type": "mid_promo_replace", "status": "REVIEW"},
            {"type": "end_card_replace", "status": "REVIEW"},
        ]
    }
    route = _production_execution_route(plan, include_review_actions=False)
    assert route["editorial"]["mid_promo_strategy"] == "skip"
    assert route["editorial"]["end_card_strategy"] == "append_own"


def test_source_context_prefers_material_bound_identity_snapshot():
    """A migrated source must not need config/brand_registry to match itself."""
    import json
    import app.branding_v09 as branding

    root = Path(tempfile.mkdtemp())
    old_workspace = branding.WORKSPACE
    try:
        branding.WORKSPACE = root
        video = root / "raw" / "run" / "ExampleApp" / "English" / "clip.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"fixture")
        identity = video.parent / "clip.mp4.identity"
        (identity / "source").mkdir(parents=True)
        (identity / "source" / "icon.png").write_bytes(b"fixture")
        profile = identity / "profile.json"
        profile.write_text(
            json.dumps(
                {
                    "brand_id": "example-app",
                    "product_name": "ExampleApp",
                    "icon": {
                        "relative_path": "raw/run/ExampleApp/English/clip.mp4.identity/source/icon.png"
                    },
                }
            ),
            encoding="utf-8",
        )
        Path(str(video) + ".metadata.json").write_text(
            json.dumps(
                {
                    "product_name": "ExampleApp",
                    "brand_id": "example-app",
                    "brand_profile": "config/brand_registry/obsolete/profile.json",
                    "brand_identity_snapshot": {
                        "profile_relative_path": "raw/run/ExampleApp/English/clip.mp4.identity/profile.json"
                    },
                }
            ),
            encoding="utf-8",
        )
        context = branding._operator_source_video_context(
            "raw/run/ExampleApp/English/clip.mp4"
        )
        assets = branding._operator_source_brand_detection_assets(context)
        assert context["brand_identity_source"] == "material_bound_snapshot"
        assert context["brand_profile_relative_path"].endswith("clip.mp4.identity/profile.json")
        assert assets["template_ready"] is True
        assert assets["icon_relative_path"].endswith("clip.mp4.identity/source/icon.png")
    finally:
        branding.WORKSPACE = old_workspace


def test_diagonal_screen_merges_one_large_word_before_counting_tiles():
    """Letters of a moving word are one mark, never four tiled marks."""
    fragments = []
    for index in range(6):
        x = 100 + index * 42
        fragments.append(
            {
                "bbox_rotated": {"x": x, "y": 320, "width": 25, "height": 44},
                "center_rotated": {"x": x + 12.5, "y": 342.0},
                "area": 25 * 44,
                "aspect_ratio": 25 / 44,
                "fill_ratio": 0.25,
                "pixel_count": int(25 * 44 * 0.25),
            }
        )
    words = _diag_merge_word_fragments(
        fragments,
        min_component_width=28,
        max_component_width=600,
        min_component_height=10,
        max_component_height=110,
        min_aspect_ratio=1.25,
        max_aspect_ratio=26.0,
    )
    assert len(words) == 1
    assert words[0]["word_fragment_count"] == 6
    assert words[0]["bbox_rotated"]["width"] == 235


def test_diagonal_visual_signature_requires_same_content():
    left = "ffffffffffffffff"
    same = left
    different = "0000000000000000"
    assert _diag_signature_similarity(left, same) == 1.0
    assert _diag_signature_similarity(left, different) < 0.2


def test_empty_legacy_router_placeholder_is_not_a_detected_watermark():
    """Zero-evidence router output must follow the no-watermark route."""
    from app.branding_v09 import _production_router_has_evidence_backed_watermark

    legacy_empty = {
        "summary": {"watermark_layer_count": 1},
        "watermark_layers": [{
            "layer_id": "unknown-watermark-layout",
            "type": "unknown",
            "handler": "calibration_required",
            "confidence": 0.0,
            "evidence": {},
        }],
    }
    assert _production_router_has_evidence_backed_watermark(legacy_empty) is False
    assert _production_router_has_evidence_backed_watermark(
        {"summary": {"watermark_layer_count": 0}, "watermark_layers": []}
    ) is False

    evidence_backed = {
        "watermark_layers": [{
            "layer_id": "census-4-unclassified-text",
            "type": "unclassified_brand_text",
            "confidence": 0.48,
            "evidence": {"bbox": {"x": 12, "y": 30, "width": 90, "height": 25}},
        }]
    }
    assert _production_router_has_evidence_backed_watermark(evidence_backed) is True


def test_router_does_not_create_an_unknown_watermark_when_all_evidence_is_empty():
    """The source router must emit a zero-layer clean result, not a fake layer."""
    import inspect
    import app.branding_v09 as branding

    router_source = inspect.getsource(branding._watermark_route_sync)
    assert 'overall_status = (\n        _overall_route_status(layers)\n        if watermark_evidence_found\n        else "NO_WATERMARK_EVIDENCE"' in router_source
    assert '"watermark_evidence_found": watermark_evidence_found' in router_source
    assert 'layers.append(\n            _layer(\n                "unknown-watermark-layout"' not in router_source


def test_source_icon_template_creates_a_fixed_lockup_candidate():
    """Stylised icon-plus-wordmark watermarks must not depend on OCR text."""
    template = np.zeros((24, 24), dtype=np.uint8)
    cv2.circle(template, (12, 12), 8, 230, thickness=-1)
    cv2.line(template, (3, 3), (20, 20), 30, thickness=2)
    frame = np.full((220, 180, 3), 70, dtype=np.uint8)
    patch = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
    frame[20:44, 16:40] = patch

    hit = _source_icon_watermark_hit(frame, template, 1.0, 180, 220)
    assert hit is not None
    assert hit["target"] == "__source_icon_watermark__"
    assert hit["bbox"]["x"] <= 18
    assert hit["bbox"]["width"] > hit["icon_bbox"]["width"]
    assert hit["bbox"]["height"] == hit["icon_bbox"]["height"]


def test_reviewed_visual_track_accepts_two_point_brief_motion():
    policy = {
        "min_observations": 3,
        "min_movement_ratio": 0.018,
        "max_interpolation_gap_seconds": 0.32,
    }
    payload = {
        "tracks": [
            {
                "track_id": "visual-dynamic-01",
                "points": [
                    {"time_seconds": 25.0, "bbox": {"x": 360, "y": 280, "width": 334, "height": 88}, "template_score": 0.24},
                    {"time_seconds": 25.25, "bbox": {"x": 374, "y": 282, "width": 334, "height": 88}, "template_score": 0.22},
                ],
            }
        ]
    }
    tracks, rejected = _strict_verified_visual_tracks(payload, 720, 1280, policy, duration=52.167)
    assert len(tracks) == 1
    assert not rejected


def test_low_score_visual_recovery_requires_strong_native_anchors():
    policy = {
        "min_observations": 3,
        "min_movement_ratio": 0.018,
        "max_interpolation_gap_seconds": 0.45,
        "min_native_edge_mean_score": 0.14,
        "min_native_edge_anchor_score": 0.20,
        "min_native_edge_anchor_count": 2,
        "min_native_edge_path_straightness": 0.45,
    }
    template = "config/dynamic_watermark_reference_reelshort_v2.png"
    payload = {
        "template_relative_paths": [template],
        "tracks": [
            {
                "track_id": "weak-scene-edge",
                "persistence_ratio": 1.0,
                "points": [
                    {"time_seconds": 30.0, "bbox": {"x": 80, "y": 900, "width": 360, "height": 80}, "template_score": 0.15, "template_relative_path": template},
                    {"time_seconds": 30.2, "bbox": {"x": 112, "y": 900, "width": 360, "height": 80}, "template_score": 0.17, "template_relative_path": template},
                    {"time_seconds": 30.4, "bbox": {"x": 144, "y": 900, "width": 360, "height": 80}, "template_score": 0.18, "template_relative_path": template},
                ],
            }
        ],
    }
    tracks, rejected = _strict_verified_visual_tracks(payload, 720, 1280, policy, duration=52.167)
    assert not tracks
    assert any(item["reason"] == "low_score_native_edge_track_lacks_strong_anchors" for item in rejected)


if __name__ == "__main__":
    test_accepts_dense_moving_verified_identity_track()
    test_rejects_two_point_or_stationary_candidate()
    test_rejects_terminal_card_track()
    test_census_strict_track_is_handed_to_renderer()
    test_same_source_diagonal_profile_returns_renderable_centers()
    test_strict_renderer_hides_outside_visibility_and_gaps()
    test_visual_detector_keeps_reviewed_wordmark_template_alongside_source_icon()
    test_dynamic_cover_is_blur_only_without_own_brand_pixels()
    test_dynamic_wordmark_blur_uses_strokes_not_whole_tracking_box()
    test_persistent_source_layout_can_be_disabled_for_dynamic_only_render()
    test_overlapping_dynamic_tracks_receive_shared_coverage_receipts()
    test_dynamic_receipt_ledger_keeps_all_resolved_tracks_despite_bottom_actions()
    test_verified_dynamic_tracks_require_renderer_handoff()
    test_fixed_layers_rebrand_while_dynamic_tracks_never_overlay()
    test_dynamic_clean_repair_rejects_unverified_fade_fallback()
    test_temporal_repair_is_skipped_when_census_has_no_verified_tracks()
    test_production_execution_route_orders_watermark_before_editorial_actions()
    test_production_route_does_not_execute_unconfirmed_review_actions()
    test_source_context_prefers_material_bound_identity_snapshot()
    test_diagonal_screen_merges_one_large_word_before_counting_tiles()
    test_diagonal_visual_signature_requires_same_content()
    test_empty_legacy_router_placeholder_is_not_a_detected_watermark()
    test_router_does_not_create_an_unknown_watermark_when_all_evidence_is_empty()
    test_source_icon_template_creates_a_fixed_lockup_candidate()
    test_reviewed_visual_track_accepts_two_point_brief_motion()
    test_low_score_visual_recovery_requires_strong_native_anchors()
    print("strict moving-watermark policy regressions passed")
