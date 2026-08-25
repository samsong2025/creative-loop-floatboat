"""Fast regression checks for Brand Registry and Watermark Engine v2 primitives.

Run from project root:
    python tools/test_watermark_engine_v2.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "api"))

from app.services.brand_identity import identity_from_detail, merge_explicit_identity  # noqa: E402
from app.services.brand_registry import BrandRegistry  # noqa: E402
from app.services.residual_qc import build_residual_report  # noqa: E402
from app.services.watermark_engine import (  # noqa: E402
    MOTION_DISAPPEAR_REAPPEAR,
    MOTION_JUMP,
    ScreenSpaceTrackBuilder,
    _candidate_score,
    _dedupe_candidates,
    _track_candidates,
    sample_times,
)
from app.models.brand import WatermarkCandidate  # noqa: E402


def test_explicit_identity_never_loses_known_product_to_null():
    merged = merge_explicit_identity(
        {"product_name": "ReelShort", "app_title": "ReelShort", "package_name": "com.newleaf.app.android.victor"},
        {"product_name": None, "app_title": None, "icon_url": "https://cdn.example/icon.png"},
    )
    assert merged["product_name"] == "ReelShort"
    assert merged["icon_url"] == "https://cdn.example/icon.png"


def test_registry_resolves_exact_package_and_preserves_icon_state():
    with tempfile.TemporaryDirectory() as temp:
        registry = BrandRegistry(Path(temp))
        identity = identity_from_detail({"product_name": "ReelShort", "app_title": "ReelShort", "package_name": "com.newleaf.app.android.victor", "icon_url": "https://cdn.example/reelshort.png"})
        saved = registry.upsert(identity)
        found = registry.resolve(package_name="com.newleaf.app.android.victor")
        assert saved["brand_id"] == "reelshort"
        assert found and found["brand_id"] == "reelshort"
        assert found["icon"]["status"] == "pending"


def test_registry_lists_and_reloads_profile_from_index_path():
    with tempfile.TemporaryDirectory() as temp:
        registry = BrandRegistry(Path(temp))
        identity = identity_from_detail({"product_name": "DramaBox", "package_name": "com.storymatrix.drama"})
        registry.upsert(identity)
        assert registry.get_profile("dramabox")["product_name"] == "DramaBox"
        assert [profile["brand_id"] for profile in registry.list_profiles()] == ["dramabox"]


def test_sparse_sampler_respects_rate_and_cap():
    assert len(sample_times(100.0, fps=2.0, cap_frames=1000)) == 201
    assert len(sample_times(1000.0, fps=10.0, cap_frames=42)) == 42


def test_temporal_tracker_builds_continuous_track_from_sparse_detections():
    candidates = [
        WatermarkCandidate(time_seconds=10.0, brand_id="reelshort", bbox=[560, 210, 125, 44], icon_similarity=0.93, detector_confidence=0.93),
        WatermarkCandidate(time_seconds=10.5, brand_id="reelshort", bbox=[554, 215, 125, 44], icon_similarity=0.91, detector_confidence=0.91),
        WatermarkCandidate(time_seconds=11.5, brand_id="reelshort", bbox=[546, 222, 125, 44], icon_similarity=0.89, detector_confidence=0.89),
    ]
    tracks = _track_candidates(candidates)
    assert len(tracks) == 1
    assert tracks[0].start_seconds == 10.0 and tracks[0].end_seconds == 11.5
    assert len(tracks[0].path) == 3 and "CONFIRMED" in tracks[0].source


def test_tracker_never_crosses_a_scene_boundary():
    candidates = [
        WatermarkCandidate(time_seconds=10.0, brand_id="reelshort", bbox=[100, 100, 60, 30], icon_similarity=0.95, detector_confidence=0.95, scene_id=0),
        WatermarkCandidate(time_seconds=10.5, brand_id="reelshort", bbox=[101, 100, 60, 30], icon_similarity=0.95, detector_confidence=0.95, scene_id=1),
    ]
    assert len(_track_candidates(candidates)) == 2


def test_roi_ocr_confirmation_joins_all_detected_words():
    words = [{"text": "Reel"}, {"text": "Short"}]
    detected = "".join("".join(str(word.get("text", "")).casefold().split()) for word in words)
    assert "reelshort" in detected


def test_same_sample_candidates_are_deduplicated_and_scored():
    candidates = [
        WatermarkCandidate(time_seconds=1.0, brand_id="reelshort", bbox=[10, 10, 100, 40], icon_similarity=0.91, detector_confidence=0.91, score=0.91),
        WatermarkCandidate(time_seconds=1.0, brand_id="reelshort", bbox=[12, 11, 100, 40], icon_similarity=0.72, detector_confidence=0.72, score=0.72),
    ]
    deduped = _dedupe_candidates(candidates)
    assert len(deduped) == 1 and deduped[0].icon_similarity == 0.91
    # Icon + detector evidence alone remains a mid-confidence candidate until
    # temporal, position, or ROI-OCR evidence is available.
    assert 0.65 < _candidate_score(deduped[0]) < 0.70


def test_tentative_tracks_are_not_formal_render_timeline_entries():
    candidates = [
        WatermarkCandidate(time_seconds=1.0, brand_id="reelshort", bbox=[10, 10, 100, 40], icon_similarity=0.91, detector_confidence=0.91, score=0.91),
    ]
    tracks = _track_candidates(candidates)
    assert len(tracks) == 1 and "TENTATIVE" in tracks[0].source


def _visual_hit(t, x, y, confidence=0.92, *, scene_id=0):
    return WatermarkCandidate(
        time_seconds=t,
        brand_id="reelshort",
        bbox=[x, y, 112, 38],
        icon_similarity=confidence,
        detector_confidence=confidence,
        score=confidence,
        scene_id=scene_id,
        source="icon_grayscale_match",
    )


def test_screen_space_builder_splits_a_jump_into_non_interpolating_segments():
    tracks = ScreenSpaceTrackBuilder(max_gap=1.25).build(
        [
            _visual_hit(18.0, 32, 48),
            _visual_hit(18.5, 34, 48),
            _visual_hit(19.0, 560, 1090),
            _visual_hit(19.5, 556, 1086),
        ],
        width=720,
        height=1280,
    )
    assert len(tracks) == 1
    assert len(tracks[0].segments) == 2
    assert tracks[0].segments[0]["end"] == 18.5
    assert tracks[0].segments[1]["start"] == 19.0
    assert tracks[0].segments[1]["motion"] == MOTION_JUMP
    assert tracks[0].segments[0]["samples"][-1]["bbox"][0] < 100
    assert tracks[0].segments[1]["samples"][0]["bbox"][0] > 500


def test_screen_space_builder_marks_disappearance_reacquisition_without_bridge():
    tracks = ScreenSpaceTrackBuilder(max_gap=0.75, max_reacquire_gap=3.0).build(
        [_visual_hit(20.0, 500, 180), _visual_hit(20.5, 495, 180), _visual_hit(22.0, 80, 620), _visual_hit(22.5, 78, 620)],
        width=720,
        height=1280,
    )
    assert len(tracks) == 1
    assert len(tracks[0].segments) == 2
    assert tracks[0].segments[1]["motion"] == MOTION_DISAPPEAR_REAPPEAR
    assert tracks[0].state == "REACQUIRED"


def test_temporal_voting_keeps_a_fading_track_confirmed():
    tracks = _track_candidates(
        [
            _visual_hit(20.0, 500, 180, 0.91),
            _visual_hit(20.5, 498, 180, 0.84),
            _visual_hit(21.0, 496, 180, 0.42),
            _visual_hit(21.5, 494, 180, 0.31),
            _visual_hit(22.0, 492, 180, 0.80),
        ],
        frame_size=(720, 1280),
    )
    assert len(tracks) == 1
    assert tracks[0].state == "CONFIRMED"
    assert "temporal_voting:CONFIRMED" in tracks[0].source


def test_scene_reset_never_merges_even_identical_screen_positions():
    tracks = ScreenSpaceTrackBuilder().build(
        [_visual_hit(10.0, 100, 100, scene_id=0), _visual_hit(10.5, 102, 100, scene_id=1)],
        width=720,
        height=1280,
    )
    assert len(tracks) == 2
    assert {track.scene_id for track in tracks} == {0, 1}


def test_residual_scan_is_a_hard_gate_with_local_intervals():
    report = build_residual_report({"brand_id": "reelshort", "candidates": [{"time_seconds": 2.0, "icon_similarity": 0.91, "source": "icon_match"}, {"time_seconds": 2.5, "icon_similarity": 0.88, "source": "icon_match"}], "metrics": {"sampled_frames": 10, "detector_runtime_ms": 20}})
    assert report["pass"] is False
    assert len(report["intervals"]) == 1
    assert report["intervals"][0]["start_seconds"] == 2.0
    assert report["intervals"][0]["end_seconds"] == 2.5
    assert report["intervals"][0]["max_confidence"] == 0.91


def test_residual_scan_accepts_edge_only_visual_evidence():
    report = build_residual_report(
        {
            "brand_id": "reelshort",
            "candidates": [{"time_seconds": 2.0, "edge_similarity": 0.91, "detector_confidence": 0.91, "bbox": [10, 20, 30, 12], "source": "edge_template_match"}],
            "metrics": {"sampled_frames": 10, "detector_runtime_ms": 20},
        }
    )
    assert report["pass"] is False
    assert report["intervals"][0]["bbox_samples"] == [[10, 20, 30, 12]]


if __name__ == "__main__":
    test_explicit_identity_never_loses_known_product_to_null()
    test_registry_resolves_exact_package_and_preserves_icon_state()
    test_registry_lists_and_reloads_profile_from_index_path()
    test_sparse_sampler_respects_rate_and_cap()
    test_temporal_tracker_builds_continuous_track_from_sparse_detections()
    test_tracker_never_crosses_a_scene_boundary()
    test_roi_ocr_confirmation_joins_all_detected_words()
    test_same_sample_candidates_are_deduplicated_and_scored()
    test_tentative_tracks_are_not_formal_render_timeline_entries()
    test_screen_space_builder_splits_a_jump_into_non_interpolating_segments()
    test_screen_space_builder_marks_disappearance_reacquisition_without_bridge()
    test_temporal_voting_keeps_a_fading_track_confirmed()
    test_scene_reset_never_merges_even_identical_screen_positions()
    test_residual_scan_is_a_hard_gate_with_local_intervals()
    test_residual_scan_accepts_edge_only_visual_evidence()
    print("watermark engine v2 regressions passed")