"""Regression checks for render safety gates introduced after V6 review.

Run from repository root with an API-compatible Python environment:
    python tools/test_branding_render_safety.py

The tests use synthetic frames and in-memory plans only. They never render,
move, overwrite or upload a source video.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))

from fastapi import HTTPException
import app.branding_v09 as branding


WIDTH, HEIGHT = 320, 180
SOURCE_BBOX = {"x": 230, "y": 8, "width": 54, "height": 18}
FULL_COVER = {"x": 222, "y": 4, "width": 72, "height": 28}


def _fixed_action(*, placement="top_right", cover=FULL_COVER, segments=None):
    action = {
        "action_id": "fixed-lockup",
        "type": "fixed_brand_overlay",
        "status": "AUTO",
        "placement": placement,
        "handler": (
            "top_cleanup_brand_overlay"
            if placement.startswith("top_")
            else "bottom_cleanup_brand_overlay"
        ),
        "bbox": dict(SOURCE_BBOX),
    }
    if cover is not None:
        action["source_full_cover_bbox"] = dict(cover)
    if segments is not None:
        action["active_segments"] = segments
    return action


def test_unconfirmed_midpromo_is_rejected_even_when_review_is_included():
    plan = {"source": {"relative_path": "raw/example.mp4"}}
    action = {
        "action_id": "detected-at-52s",
        "type": "mid_promo_replace",
        "status": "REVIEW",
        "start_seconds": 52.0,
        "end_seconds": 55.0,
        "source_candidate": {"requires_human_review": True},
    }
    try:
        branding._operator_validate_editorial_actions_for_render(
            plan, [action], include_review_actions=True
        )
    except HTTPException as exc:
        assert exc.status_code == 422
        assert exc.detail["error"] == "mid_promo_requires_explicit_user_confirmation"
    else:
        raise AssertionError("unconfirmed in-story replacement must be rejected")


def test_source_bound_confirmed_midpromo_is_allowed():
    source_sha256 = "a" * 64
    action = {
        "action_id": "confirmed-boundary",
        "type": "mid_promo_replace",
        "status": "AUTO",
        "start_seconds": 52.0,
        "end_seconds": 55.0,
        "decision_source": "user_review_confirmed_boundary",
        "source_candidate": {"user_confirmed": True, "source_sha256": source_sha256},
    }
    branding._operator_validate_editorial_actions_for_render(
        {"source": {"relative_path": "raw/example.mp4", "sha256": source_sha256}},
        [action],
        include_review_actions=False,
    )


def test_midpromo_confirmation_for_a_different_source_is_rejected():
    action = {
        "action_id": "wrong-source-boundary",
        "type": "mid_promo_replace",
        "status": "AUTO",
        "start_seconds": 52.0,
        "end_seconds": 55.0,
        "decision_source": "user_review_confirmed_boundary",
        "source_candidate": {"user_confirmed": True, "source_sha256": "b" * 64},
    }
    try:
        branding._operator_validate_editorial_actions_for_render(
            {"source": {"relative_path": "raw/example.mp4", "sha256": "a" * 64}},
            [action],
            include_review_actions=False,
        )
    except HTTPException as exc:
        assert exc.status_code == 422
        assert exc.detail["error"] == "mid_promo_requires_explicit_user_confirmation"
    else:
        raise AssertionError("a confirmation for a different source must be rejected")


def test_fixed_cover_requires_explicit_geometry_not_detector_bbox():
    action = _fixed_action(cover=None)
    assert branding._operator_fixed_full_cover_bbox(action, WIDTH, HEIGHT) is None
    try:
        branding._operator_validate_fixed_cover_actions([action], WIDTH, HEIGHT)
    except HTTPException as exc:
        assert exc.status_code == 422
        assert exc.detail["error"] == "fixed_watermark_full_cover_geometry_required"
    else:
        raise AssertionError("detector bbox must not masquerade as full cover")


def test_fixed_cover_is_used_by_top_and_bottom_render_paths():
    top = _fixed_action()
    assert branding._operator_fixed_action_has_full_cover_contract(top, WIDTH, HEIGHT)
    assert branding._operator_fixed_full_cover_bbox(top, WIDTH, HEIGHT) == FULL_COVER

    bottom_cover = {"x": 22, "y": 144, "width": 88, "height": 28}
    bottom = _fixed_action(
        placement="bottom_left",
        cover=bottom_cover,
        segments=[
            {
                "start_seconds": 0.0,
                "end_seconds": 5.0,
                "bbox": {"x": 30, "y": 150, "width": 58, "height": 16},
                "source_full_cover_bbox": bottom_cover,
            }
        ],
    )
    assert branding._operator_bottom_action_bbox_at_time(
        bottom, 2.0, 5.0, WIDTH, HEIGHT
    ) == bottom_cover


def test_blank_brand_asset_is_rejected_before_composition():
    root = Path(tempfile.mkdtemp())
    assets = root / "assets" / "brand" / "floatboat-fde"
    assets.mkdir(parents=True)
    original_workspace = branding.WORKSPACE
    try:
        branding.WORKSPACE = root
        for filename in ("icon.png", "logo.png"):
            path = assets / filename
            assert cv2.imwrite(str(path), np.full((12, 12, 3), 128, dtype=np.uint8))
            result = branding._validate_image_asset(
                f"assets/brand/floatboat-fde/{filename}"
            )
            assert result["valid"] is False
            assert result["error"] == "image_visual_content_blank"
    finally:
        branding.WORKSPACE = original_workspace


def test_top_renderer_receipts_actual_full_cover_geometry():
    # Avoid dependence on real brand art: text-only mode exercises the source
    # cleanup/opaque plate and verifies that the cover receipt is the contract.
    frame = np.full((HEIGHT, WIDTH, 3), 110, dtype=np.uint8)
    frame[4:32, 222:294] = (20, 20, 220)
    request = SimpleNamespace(
        watermark_treatment_mode="rebrand_fixed_clean_dynamic",
        compose_top_brand_layers=True,
        top_brand_mode="text_only",
        top_brand_span_guard_seconds=0.0,
        top_text_cleanup_expand_x_ratio=0.0,
        top_text_cleanup_expand_y_ratio=0.0,
        top_icon_strong_cover_enabled=True,
    )
    profile = {"product_name": "Floatboat", "brand_name": "Floatboat", "assets": {}}
    action = _fixed_action(
        segments=[
            {
                "start_seconds": 0.0,
                "end_seconds": 5.0,
                "bbox": dict(SOURCE_BBOX),
                "source_full_cover_bbox": dict(FULL_COVER),
            }
        ]
    )
    _, applied = branding._apply_top_brand_composition_render(
        frame, 1.0, [action], profile, WIDTH, HEIGHT, 5.0, request
    )
    assert len(applied) == 1
    assert applied[0]["source_cover_bbox"] == FULL_COVER


def main():
    test_unconfirmed_midpromo_is_rejected_even_when_review_is_included()
    test_source_bound_confirmed_midpromo_is_allowed()
    test_midpromo_confirmation_for_a_different_source_is_rejected()
    test_fixed_cover_requires_explicit_geometry_not_detector_bbox()
    test_fixed_cover_is_used_by_top_and_bottom_render_paths()
    test_blank_brand_asset_is_rejected_before_composition()
    test_top_renderer_receipts_actual_full_cover_geometry()
    print("branding render safety regressions passed")


if __name__ == "__main__":
    main()
