from __future__ import annotations

"""Dynamic Watermark Engine v1: visual anchors -> screen-space tracks.

OCR is deliberately a candidate-ROI confirmation signal, never the primary
detector.  The renderer receives one canonical track format with hard segment
boundaries so a screen-overlay JUMP can never be interpolated across a scene.
"""

import time
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from ..models.brand import DynamicWatermarkTrack, WatermarkCandidate, WatermarkTrack


DEFAULT_SCORE_WEIGHTS = {"icon": 0.45, "detector": 0.30, "text": 0.10, "temporal": 0.10, "position": 0.05}

MOTION_CONTINUOUS = "continuous"
MOTION_STATIONARY = "stationary"
MOTION_JUMP = "jump"
MOTION_DISAPPEAR_REAPPEAR = "disappear_reappear"

STATE_TENTATIVE = "TENTATIVE"
STATE_CONFIRMED = "CONFIRMED"
STATE_FADING = "FADING"
STATE_COASTING = "COASTING"
STATE_REACQUIRED = "REACQUIRED"
STATE_LOST = "LOST"


def sample_times(duration: float, fps: float = 2.0, cap_frames: int = 1000) -> list[float]:
    """Return low-frequency detector timestamps, including the video end."""
    fps = max(0.1, float(fps))
    count = min(max(1, int(duration * fps) + 1), max(1, int(cap_frames)))
    return [min(float(duration), index / fps) for index in range(count)]


def _score(parts: dict[str, float], weights: dict[str, float] | None = None) -> float:
    weights = {**DEFAULT_SCORE_WEIGHTS, **(weights or {})}
    return max(0.0, min(1.0, sum(float(parts.get(key, 0.0)) * weight for key, weight in weights.items())))


def _candidate_score(candidate: WatermarkCandidate, weights: dict[str, float] | None = None) -> float:
    """Fuse visual, ROI-OCR and temporal evidence under the compatible v2 API."""
    return _score(
        {
            "icon": candidate.icon_similarity,
            "detector": candidate.detector_confidence,
            "text": candidate.ocr_brand_match,
            "temporal": candidate.temporal_consistency,
            "position": candidate.spatial_prior,
        },
        weights,
    )


def _effective_score(candidate: WatermarkCandidate) -> float:
    return float(candidate.score) if float(candidate.score) > 0 else _candidate_score(candidate)


def _iou(a: list[float], b: list[float]) -> float:
    ax, ay, aw, ah = (float(v) for v in a)
    bx, by, bw, bh = (float(v) for v in b)
    left, top, right, bottom = max(ax, bx), max(ay, by), min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    return intersection / max(1.0, aw * ah + bw * bh - intersection)


def _center(bbox: list[float]) -> tuple[float, float]:
    x, y, width, height = (float(value) for value in bbox)
    return x + width / 2.0, y + height / 2.0


def _merge_candidate_evidence(candidates: list[WatermarkCandidate], iou_floor: float = 0.20) -> list[WatermarkCandidate]:
    """Fuse gray/edge hits for the same sampled icon instead of double tracking."""
    output: list[WatermarkCandidate] = []
    for candidate in sorted(candidates, key=_effective_score, reverse=True):
        match = next(
            (
                existing
                for existing in output
                if existing.brand_id == candidate.brand_id
                and existing.scene_id == candidate.scene_id
                and abs(existing.time_seconds - candidate.time_seconds) <= 1e-6
                and _iou(existing.bbox, candidate.bbox) >= iou_floor
            ),
            None,
        )
        if match is None:
            output.append(candidate)
            continue
        replacement = _effective_score(candidate) > _effective_score(match)
        if replacement:
            match.bbox, match.scale_factor = list(candidate.bbox), candidate.scale_factor
        match.icon_similarity = max(match.icon_similarity, candidate.icon_similarity)
        match.edge_similarity = max(match.edge_similarity, candidate.edge_similarity)
        match.detector_confidence = max(match.detector_confidence, candidate.detector_confidence)
        match.spatial_prior = max(match.spatial_prior, candidate.spatial_prior)
        match.score = max(_effective_score(match), _effective_score(candidate))
        match.source = "+".join(sorted(set(str(match.source).split("+")) | set(str(candidate.source).split("+"))))
    return sorted(output, key=lambda item: item.time_seconds)


def _dedupe_candidates(candidates: list[WatermarkCandidate], iou_floor: float = 0.45) -> list[WatermarkCandidate]:
    """Keep only the strongest overlapping detection at each sampled time."""
    output: list[WatermarkCandidate] = []
    for candidate in sorted(candidates, key=_effective_score, reverse=True):
        if any(
            abs(candidate.time_seconds - existing.time_seconds) < 1e-6
            and candidate.brand_id == existing.brand_id
            and candidate.scene_id == existing.scene_id
            and _iou(candidate.bbox, existing.bbox) >= iou_floor
            for existing in output
        ):
            continue
        output.append(candidate)
    return sorted(output, key=lambda item: item.time_seconds)


def _classify_motion(
    previous: WatermarkCandidate | dict[str, Any],
    current: WatermarkCandidate | dict[str, Any],
    width: int,
    height: int,
    *,
    max_gap: float = 1.25,
) -> str:
    """Identify overlay motion. JUMP/reappear is always a hard segment break."""
    previous_time = float(previous.time_seconds if isinstance(previous, WatermarkCandidate) else previous.get("t", 0.0))
    current_time = float(current.time_seconds if isinstance(current, WatermarkCandidate) else current.get("t", 0.0))
    previous_bbox = list(previous.bbox if isinstance(previous, WatermarkCandidate) else previous.get("bbox", []))
    current_bbox = list(current.bbox if isinstance(current, WatermarkCandidate) else current.get("bbox", []))
    if len(previous_bbox) != 4 or len(current_bbox) != 4 or current_time - previous_time > max_gap + 1e-6:
        return MOTION_DISAPPEAR_REAPPEAR
    px, py = _center(previous_bbox)
    cx, cy = _center(current_bbox)
    normalized_distance = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5 / max(1.0, float(min(width, height)))
    if normalized_distance >= 0.12 or (_iou(previous_bbox, current_bbox) < 0.02 and normalized_distance >= 0.08):
        return MOTION_JUMP
    return MOTION_STATIONARY if normalized_distance <= 0.008 else MOTION_CONTINUOUS


def _path_node(candidate: WatermarkCandidate) -> dict[str, Any]:
    return {
        "t": round(float(candidate.time_seconds), 4),
        "bbox": [round(float(value), 3) for value in candidate.bbox],
        "observed": bool(candidate.observed),
        "scene_id": int(candidate.scene_id),
        "score": round(_effective_score(candidate), 4),
        "icon_similarity": round(float(candidate.icon_similarity), 4),
        "edge_similarity": round(float(candidate.edge_similarity), 4),
        "ocr_brand_match": round(float(candidate.ocr_brand_match), 4),
        "source": str(candidate.source),
    }


def _segment_payload(
    samples: list[WatermarkCandidate],
    width: int,
    height: int,
    *,
    max_gap: float,
    incoming_motion: str | None = None,
) -> dict[str, Any]:
    internal_motions = [_classify_motion(a, b, width, height, max_gap=max_gap) for a, b in zip(samples, samples[1:])]
    if incoming_motion:
        motion = incoming_motion
    elif MOTION_CONTINUOUS in internal_motions:
        motion = MOTION_CONTINUOUS
    else:
        motion = MOTION_STATIONARY
    return {
        "motion": motion,
        "start": round(float(samples[0].time_seconds), 4),
        "end": round(float(samples[-1].time_seconds), 4),
        "samples": [_path_node(item) for item in samples],
        # The incoming JUMP labels the segment but does not stop interpolation
        # *inside* the new segment once that segment has two continuous samples.
        "interpolation_allowed": len(samples) >= 2 and all(item in (MOTION_CONTINUOUS, MOTION_STATIONARY) for item in internal_motions),
        "max_interpolation_gap_seconds": float(max_gap),
    }


def _build_segments(path: list[WatermarkCandidate], width: int, height: int, *, max_gap: float = 1.25) -> list[dict[str, Any]]:
    ordered = sorted(path, key=lambda item: item.time_seconds)
    if not ordered:
        return []
    segments: list[dict[str, Any]] = []
    current = [ordered[0]]
    incoming_motion: str | None = None
    for previous, candidate in zip(ordered, ordered[1:]):
        motion = _classify_motion(previous, candidate, width, height, max_gap=max_gap)
        if motion in (MOTION_JUMP, MOTION_DISAPPEAR_REAPPEAR):
            segments.append(_segment_payload(current, width, height, max_gap=max_gap, incoming_motion=incoming_motion))
            current, incoming_motion = [candidate], motion
        else:
            current.append(candidate)
    segments.append(_segment_payload(current, width, height, max_gap=max_gap, incoming_motion=incoming_motion))
    return segments


def _temporal_vote(path: list[WatermarkCandidate], *, window_size: int = 5, threshold: float = 0.55) -> int:
    scores = [_effective_score(item) for item in sorted(path, key=lambda item: item.time_seconds)]
    return max((sum(score >= threshold for score in scores[index : index + window_size]) for index in range(len(scores))), default=0)


def _track_state(path: list[WatermarkCandidate], *, max_gap: float) -> str:
    ordered = sorted(path, key=lambda item: item.time_seconds)
    if not ordered:
        return STATE_LOST
    scores = [_effective_score(item) for item in ordered]
    high_count = sum(
        max(item.icon_similarity, item.edge_similarity, item.detector_confidence) >= 0.82
        for item in ordered
    )
    support = any(
        abs(a.time_seconds - b.time_seconds) <= 1.0 and _effective_score(a) >= 0.55 and _effective_score(b) >= 0.55
        for index, a in enumerate(ordered)
        for b in ordered[index + 1 :]
    )
    confirmed = sum(scores) / len(scores) >= 0.52 and (_temporal_vote(ordered) >= 3 or (high_count >= 1 and support))
    if not confirmed:
        return STATE_TENTATIVE
    # A confirmed track that survived an earlier visibility gap is explicitly
    # REACQUIRED even if it now has a few normal samples after re-entry.
    had_visibility_gap = any(
        right.time_seconds - left.time_seconds > max_gap + 1e-6
        for left, right in zip(ordered, ordered[1:])
    )
    if had_visibility_gap:
        return STATE_REACQUIRED if _effective_score(ordered[-1]) >= 0.70 else STATE_COASTING
    return STATE_FADING if _effective_score(ordered[-1]) < 0.52 else STATE_CONFIRMED


def _track_candidates(
    candidates: list[WatermarkCandidate],
    max_gap: float = 1.25,
    iou_floor: float = 0.12,
    *,
    frame_size: tuple[int, int] | None = None,
    max_reacquire_gap: float = 3.0,
) -> list[WatermarkTrack]:
    """Screen-Space Track Builder with scene-cut reset and re-acquisition gaps."""
    width, height = frame_size or (1280, 720)
    raw_tracks: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: item.time_seconds):
        choices: list[tuple[float, dict[str, Any]]] = []
        for track in raw_tracks:
            if track["brand_id"] != candidate.brand_id or track["scene_id"] != candidate.scene_id:
                continue
            previous = track["path"][-1]
            delta = candidate.time_seconds - track["last_time"]
            if delta <= 1e-6 or delta > max_reacquire_gap:
                continue
            compatible = _iou(previous.bbox, candidate.bbox) >= iou_floor or delta <= 0.55
            if delta > max_gap:
                # Re-acquisition is based on raw visual evidence, not the
                # conservative fused score (which intentionally discounts a
                # single observation with no temporal/OCR support).
                previous_visual = max(previous.icon_similarity, previous.edge_similarity, previous.detector_confidence)
                candidate_visual = max(candidate.icon_similarity, candidate.edge_similarity, candidate.detector_confidence)
                compatible = previous_visual >= 0.65 and candidate_visual >= 0.70
            if compatible:
                px, py = _center(previous.bbox)
                cx, cy = _center(candidate.bbox)
                choices.append((delta + ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5 / max(1.0, min(width, height)), track))
        if choices:
            chosen = min(choices, key=lambda item: item[0])[1]
        else:
            chosen = {"brand_id": candidate.brand_id, "scene_id": candidate.scene_id, "last_time": candidate.time_seconds, "path": []}
            raw_tracks.append(chosen)
        chosen["last_time"] = candidate.time_seconds
        chosen["path"].append(candidate)

    output: list[DynamicWatermarkTrack] = []
    for index, track in enumerate(raw_tracks, 1):
        path = sorted(track["path"], key=lambda item: item.time_seconds)
        for item in path:
            support = sum(1 for other in path if abs(other.time_seconds - item.time_seconds) <= 2.0 and _effective_score(other) >= 0.55)
            item.temporal_consistency = min(1.0, support / 3.0)
            item.score = _candidate_score(item)
        state = _track_state(path, max_gap=max_gap)
        segments = _build_segments(path, width, height, max_gap=max_gap)
        confidence = sum(_effective_score(item) for item in path) / len(path)
        sources = sorted({part for item in path for part in str(item.source).split("+") if part})
        output.append(
            DynamicWatermarkTrack(
                track_id=f"{track['brand_id']}-dw-{index:03d}",
                brand_id=track["brand_id"],
                start_seconds=float(path[0].time_seconds),
                end_seconds=float(path[-1].time_seconds),
                confidence=max(0.0, min(1.0, confidence)),
                source="+".join([*sources, f"temporal_voting:{state}"]),
                state=state,
                scene_id=int(track["scene_id"]),
                motion_summary=list(dict.fromkeys(str(segment["motion"]) for segment in segments)),
                anchor_count=len(path),
                segments=segments,
                path=[_path_node(item) for item in path],
            )
        )
    return output


def _load_image(path: Path):
    try:
        return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    except Exception:
        return None


def _profile_templates(video_path: Path, profile: dict[str, Any]) -> dict[str, Any]:
    workspace = Path(str(profile.get("_workspace_dir"))) if profile.get("_workspace_dir") else next((parent for parent in video_path.parents if parent.name == "workspace"), video_path.parent)
    icon_path = (profile.get("icon") or {}).get("relative_path")
    assets = (profile.get("templates") or {}).get("assets") or {}
    icon = _load_image(workspace / str(icon_path)) if icon_path else None
    gray = _load_image(workspace / str(assets["icon-gray.png"])) if assets.get("icon-gray.png") else None
    if gray is None:
        gray = icon
    edge = _load_image(workspace / str(assets["icon-edge.png"])) if assets.get("icon-edge.png") else None
    edge_dilated = _load_image(workspace / str(assets["icon-edge-dilated.png"])) if assets.get("icon-edge-dilated.png") else None
    if edge is None and gray is not None and gray.size:
        edge = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 64, 160)
    if edge_dilated is None and edge is not None and edge.size:
        edge_dilated = cv2.dilate(edge, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
    return {"gray": gray, "edge": edge, "edge_dilated": edge_dilated}


def _template_candidates(frame, template, *, kind: str, threshold: float, brand_id: str, second: float, scene_id: int) -> list[WatermarkCandidate]:
    if template is None or getattr(template, "size", 0) == 0:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    is_edge = kind.startswith("edge")
    search = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 64, 160) if is_edge else gray
    local_threshold = max(0.48, threshold - 0.12) if is_edge else threshold
    hits: list[WatermarkCandidate] = []
    for scale in (0.50, 0.67, 0.80, 1.0, 1.25, 1.50):
        width, height = max(8, int(template.shape[1] * scale)), max(8, int(template.shape[0] * scale))
        if width >= search.shape[1] or height >= search.shape[0]:
            continue
        resized = cv2.resize(template, (width, height), interpolation=cv2.INTER_AREA)
        if float(resized.std()) < 1.0:
            continue
        result = cv2.matchTemplate(search, resized, cv2.TM_CCOEFF_NORMED)
        _, value, _, location = cv2.minMaxLoc(result)
        if float(value) < local_threshold:
            continue
        x, y = location
        spatial = 1.0 if y < search.shape[0] * 0.35 or y > search.shape[0] * 0.65 else 0.85
        candidate = WatermarkCandidate(
            time_seconds=second,
            brand_id=brand_id,
            bbox=[float(x), float(y), float(width), float(height)],
            icon_similarity=float(value) if not is_edge else 0.0,
            edge_similarity=float(value) if is_edge else 0.0,
            scale_factor=float(scale),
            detector_confidence=float(value),
            spatial_prior=spatial,
            scene_id=scene_id,
            source="edge_template_match" if is_edge else "icon_grayscale_match",
        )
        candidate.score = _candidate_score(candidate)
        hits.append(candidate)
    return sorted(hits, key=_effective_score, reverse=True)[:4]


def _visual_candidates(frame, templates: dict[str, Any], *, threshold: float, brand_id: str, second: float, scene_id: int) -> list[WatermarkCandidate]:
    hits: list[WatermarkCandidate] = []
    hits += _template_candidates(frame, templates.get("gray"), kind="gray", threshold=threshold, brand_id=brand_id, second=second, scene_id=scene_id)
    hits += _template_candidates(frame, templates.get("edge"), kind="edge", threshold=threshold, brand_id=brand_id, second=second, scene_id=scene_id)
    hits += _template_candidates(frame, templates.get("edge_dilated"), kind="edge_dilated", threshold=threshold, brand_id=brand_id, second=second, scene_id=scene_id)
    return _merge_candidate_evidence(hits)


def _icon_candidates(frame, icon_gray, threshold: float, brand_id: str, second: float, scene_id: int) -> list[WatermarkCandidate]:
    """Compatibility hook retained for callers that have only an icon template."""
    return _template_candidates(frame, icon_gray, kind="gray", threshold=threshold, brand_id=brand_id, second=second, scene_id=scene_id)


def _roi_ocr_confirmation(frame, candidates: list[WatermarkCandidate], aliases: list[str]) -> list[WatermarkCandidate]:
    """Use OCR only inside icon/edge candidate ROIs; failure remains non-fatal."""
    aliases = [str(alias).casefold().replace(" ", "") for alias in aliases if str(alias).strip()]
    if not candidates or not aliases:
        return candidates
    try:
        try:
            from .. import branding_v09 as legacy_branding
        except ImportError:  # pragma: no cover
            import branding_v09 as legacy_branding
    except Exception:
        return candidates
    frame_height, frame_width = frame.shape[:2]
    for candidate in candidates:
        x, y, box_width, box_height = candidate.bbox
        pad_x, pad_y = max(4, int(box_width * 0.20)), max(4, int(box_height * 0.35))
        left, top = max(0, int(x) - pad_x), max(0, int(y) - pad_y)
        right, bottom = min(frame_width, int(x + box_width) + pad_x), min(frame_height, int(y + box_height) + pad_y)
        roi = frame[top:bottom, left:right]
        if roi.size == 0:
            continue
        try:
            words = legacy_branding._tesseract_words(roi, "eng", 45.0, 1.5)
        except Exception:
            continue
        detected = "".join("".join(str(word.get("text", "")).casefold().split()) for word in words)
        if any(alias in detected for alias in aliases):
            candidate.ocr_brand_match = 1.0
            candidate.source = "+".join(dict.fromkeys([*str(candidate.source).split("+"), "roi_ocr_confirmation"]))
            candidate.score = _candidate_score(candidate)
    return candidates


def _is_scene_cut(previous_histogram, previous_gray, gray) -> bool:
    if previous_histogram is None or previous_gray is None:
        return False
    histogram = cv2.calcHist([gray], [0], None, [32], [0, 256])
    cv2.normalize(histogram, histogram)
    return cv2.compareHist(previous_histogram, histogram, cv2.HISTCMP_BHATTACHARYYA) >= 0.38 or float(cv2.absdiff(previous_gray, gray).mean()) >= 48.0


def _clamp_bbox(bbox: list[float], width: int, height: int) -> list[float]:
    x, y, box_width, box_height = (float(value) for value in bbox)
    box_width = max(2.0, min(box_width, float(width)))
    box_height = max(2.0, min(box_height, float(height)))
    return [max(0.0, min(x, width - box_width)), max(0.0, min(y, height - box_height)), box_width, box_height]


def _local_template_refinement(gray, template, predicted_bbox: list[float], width: int, height: int) -> tuple[list[float] | None, float]:
    """Refine an optical-flow/Kalman prediction inside a deliberately local ROI."""
    if template is None or template.size == 0:
        return None, 0.0
    x, y, box_width, box_height = _clamp_bbox(predicted_bbox, width, height)
    radius_x, radius_y = max(8, int(box_width * 0.75)), max(8, int(box_height * 0.75))
    left, top = max(0, int(x) - radius_x), max(0, int(y) - radius_y)
    right, bottom = min(width, int(x + box_width) + radius_x), min(height, int(y + box_height) + radius_y)
    roi = gray[top:bottom, left:right]
    if roi.shape[0] < template.shape[0] or roi.shape[1] < template.shape[1] or float(template.std()) < 1.0:
        return None, 0.0
    result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, location = cv2.minMaxLoc(result)
    if float(score) < 0.35:
        return None, float(score)
    return _clamp_bbox([float(left + location[0]), float(top + location[1]), float(template.shape[1]), float(template.shape[0])], width, height), float(score)


def _dense_samples_for_continuous_segment(video_path: Path, segment: dict[str, Any], width: int, height: int, fps: float) -> list[dict[str, Any]]:
    """Fill a CONTINUOUS segment at video FPS using flow + Kalman + local template.

    Detector anchors remain the source of truth and are periodically injected as
    Kalman measurements.  There is no extrapolation past a segment boundary,
    and failure falls back to its sparse anchor samples instead of guessing.
    """
    anchors = sorted([row for row in segment.get("samples", []) if len(row.get("bbox", [])) == 4], key=lambda row: float(row.get("t", 0.0)))
    if len(anchors) < 2 or not bool(segment.get("interpolation_allowed")):
        return anchors
    start_frame = max(0, int(round(float(anchors[0]["t"]) * fps)))
    end_frame = max(start_frame, int(round(float(anchors[-1]["t"]) * fps)))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return anchors
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ok, first_frame = cap.read()
        if not ok or first_frame is None:
            return anchors
        previous_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        initial_bbox = _clamp_bbox([float(value) for value in anchors[0]["bbox"]], width, height)
        ix, iy, iw, ih = (int(round(value)) for value in initial_bbox)
        template = previous_gray[iy : iy + ih, ix : ix + iw].copy()
        if template.size == 0:
            return anchors
        kalman = cv2.KalmanFilter(4, 2)
        dt = 1.0 / max(1e-6, fps)
        kalman.transitionMatrix = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.12
        kalman.errorCovPost = np.eye(4, dtype=np.float32)
        cx, cy = _center(initial_bbox)
        kalman.statePost = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
        previous_point = np.array([[[cx, cy]]], dtype=np.float32)
        anchor_by_frame = {int(round(float(anchor["t"]) * fps)): anchor for anchor in anchors}
        output: list[dict[str, Any]] = []
        current_bbox = initial_bbox
        for frame_index in range(start_frame, end_frame + 1):
            second = frame_index / max(1e-6, fps)
            anchor = anchor_by_frame.get(frame_index)
            if frame_index == start_frame:
                gray = previous_gray
            else:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                prediction = kalman.predict()
                predicted_center = (float(prediction[0, 0]), float(prediction[1, 0]))
                predicted_bbox = _clamp_bbox([predicted_center[0] - current_bbox[2] / 2, predicted_center[1] - current_bbox[3] / 2, current_bbox[2], current_bbox[3]], width, height)
                flow_point, status, _ = cv2.calcOpticalFlowPyrLK(previous_gray, gray, previous_point, None, winSize=(21, 21), maxLevel=3)
                if status is not None and int(status[0, 0]) == 1:
                    flow_center = (float(flow_point[0, 0, 0]), float(flow_point[0, 0, 1]))
                    predicted_bbox[0] = flow_center[0] - predicted_bbox[2] / 2
                    predicted_bbox[1] = flow_center[1] - predicted_bbox[3] / 2
                refined_bbox, refinement_score = _local_template_refinement(gray, template, predicted_bbox, width, height)
                if refined_bbox is not None:
                    predicted_bbox = refined_bbox
                current_bbox = _clamp_bbox(predicted_bbox, width, height)
                previous_gray = gray
            if anchor is not None:
                current_bbox = _clamp_bbox([float(value) for value in anchor["bbox"]], width, height)
                observed, source, score = True, str(anchor.get("source", "anchor")), float(anchor.get("score", 0.0))
            else:
                observed, source, score = False, "kalman_prediction+optical_flow+local_template_refinement", 0.0
            mx, my = _center(current_bbox)
            kalman.correct(np.array([[mx], [my]], dtype=np.float32))
            previous_point = np.array([[[mx, my]]], dtype=np.float32)
            output.append(
                {
                    "t": round(second, 4),
                    "bbox": [round(float(value), 3) for value in current_bbox],
                    "observed": observed,
                    "scene_id": int(anchors[0].get("scene_id", 0)),
                    "score": round(score, 4),
                    "source": source,
                }
            )
        return output or anchors
    except cv2.error:
        return anchors
    finally:
        cap.release()


def _densify_continuous_tracks(video_path: Path, tracks: list[WatermarkTrack], width: int, height: int, fps: float) -> int:
    """Run dense tracker only for continuous segments; returns generated samples."""
    generated = 0
    for track in tracks:
        for segment in track.segments:
            if str(segment.get("motion")) != MOTION_CONTINUOUS:
                continue
            original = list(segment.get("samples") or [])
            dense = _dense_samples_for_continuous_segment(video_path, segment, width, height, fps)
            if len(dense) > len(original):
                segment["samples"] = dense
                segment["dense_tracking"] = {"enabled": True, "method": "kalman_prediction+optical_flow+local_template_refinement", "fps": round(float(fps), 4), "generated_samples": len(dense) - len(original)}
                generated += len(dense) - len(original)
    return generated


def _profile_aliases(profile: dict[str, Any]) -> list[str]:
    """Add the explicit product token (e.g. ReelShort) to detail-title aliases."""
    aliases = [str(value) for value in profile.get("aliases", []) if str(value).strip()]
    for field in ("product_name", "app_title"):
        value = str(profile.get(field) or "").strip()
        if value:
            aliases.append(value)
            short = value.split(" - ", 1)[0].strip()
            if short:
                aliases.append(short)
    return list(dict.fromkeys(aliases))


def _verify_strong_singletons(
    video_path: Path,
    candidates: list[WatermarkCandidate],
    templates: dict[str, Any],
    profile: dict[str, Any],
    *,
    width: int,
    height: int,
    threshold: float,
    enable_ocr_confirmation: bool,
) -> tuple[list[WatermarkCandidate], int]:
    """Perform the v1 tentative-track look-back/look-ahead confirmation pass.

    A lone high-confidence icon match remains a TENTATIVE track, but is not
    discarded.  We inspect its ±0.5/±1.0s neighborhood at most once per target
    timestamp.  A material scene change rejects that local sample so a new
    scene cannot accidentally supply confirmation evidence for the old one.
    """
    original = list(candidates)
    strong = [
        candidate
        for candidate in original
        if max(float(candidate.icon_similarity), float(candidate.edge_similarity), float(candidate.detector_confidence)) >= 0.86
        and not any(
            other is not candidate
            and other.brand_id == candidate.brand_id
            and other.scene_id == candidate.scene_id
            and 0.0 < abs(other.time_seconds - candidate.time_seconds) <= 0.75
            for other in original
        )
    ]
    if not strong:
        return candidates, 0
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return candidates, 0
    added: list[WatermarkCandidate] = []
    inspected: set[float] = set()
    ocr_rois = 0
    try:
        for anchor in strong:
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, anchor.time_seconds) * 1000.0)
            ok, anchor_frame = cap.read()
            if not ok or anchor_frame is None:
                continue
            anchor_gray = cv2.cvtColor(anchor_frame, cv2.COLOR_BGR2GRAY)
            anchor_histogram = cv2.calcHist([anchor_gray], [0], None, [32], [0, 256])
            cv2.normalize(anchor_histogram, anchor_histogram)
            for offset in (-1.0, -0.5, 0.5, 1.0):
                second = round(max(0.0, anchor.time_seconds + offset), 4)
                if second in inspected or any(abs(row.time_seconds - second) <= 0.05 for row in [*original, *added]):
                    continue
                inspected.add(second)
                cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000.0)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if _is_scene_cut(anchor_histogram, anchor_gray, gray):
                    continue
                nearby = _visual_candidates(frame, templates, threshold=threshold, brand_id=anchor.brand_id, second=second, scene_id=anchor.scene_id)
                for item in nearby:
                    item.source = "+".join(dict.fromkeys([*str(item.source).split("+"), "tentative_back_forward_verification"]))
                if enable_ocr_confirmation and nearby:
                    checked = nearby[:2]
                    ocr_rois += len(checked)
                    _roi_ocr_confirmation(frame, checked, _profile_aliases(profile))
                added.extend(nearby)
    finally:
        cap.release()
    return _dedupe_candidates(_merge_candidate_evidence([*candidates, *added]), iou_floor=0.35), ocr_rois


def analyze_video(
    video_path: Path,
    profile: dict[str, Any],
    *,
    detector_fps: float = 2.0,
    threshold: float = 0.72,
    max_frames: int = 1000,
    weights: dict[str, float] | None = None,
    enable_ocr_confirmation: bool = True,
    max_reacquire_gap_seconds: float = 3.0,
) -> dict[str, Any]:
    """Low-frequency visual scan (2–4 FPS by default) -> canonical tracks."""
    started = time.perf_counter()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    duration = frame_count / fps if frame_count else 0.0
    brand_id = str(profile["brand_id"])
    templates = _profile_templates(video_path, profile)
    candidates: list[WatermarkCandidate] = []
    previous_histogram = previous_gray = None
    scene_id = ocr_rois = 0
    times = sample_times(duration, detector_fps, max_frames)
    for second in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if _is_scene_cut(previous_histogram, previous_gray, gray):
            scene_id += 1
        histogram = cv2.calcHist([gray], [0], None, [32], [0, 256])
        cv2.normalize(histogram, histogram)
        previous_histogram, previous_gray = histogram, gray
        frame_candidates = _visual_candidates(frame, templates, threshold=threshold, brand_id=brand_id, second=second, scene_id=scene_id)
        if enable_ocr_confirmation and frame_candidates:
            checked = frame_candidates[:3]
            ocr_rois += len(checked)
            _roi_ocr_confirmation(frame, checked, _profile_aliases(profile))
        candidates.extend(frame_candidates)
    cap.release()
    candidates = _dedupe_candidates(_merge_candidate_evidence(candidates), iou_floor=0.35)
    candidates, tentative_verification_ocr_rois = _verify_strong_singletons(
        video_path,
        candidates,
        templates,
        profile,
        width=width,
        height=height,
        threshold=threshold,
        enable_ocr_confirmation=enable_ocr_confirmation,
    )
    ocr_rois += tentative_verification_ocr_rois
    all_tracks = _track_candidates(candidates, frame_size=(width, height), max_reacquire_gap=max(0.5, float(max_reacquire_gap_seconds)))
    dense_tracker_samples = _densify_continuous_tracks(video_path, all_tracks, width, height, fps)
    tracks = [track for track in all_tracks if track.state in {STATE_CONFIRMED, STATE_FADING, STATE_COASTING, STATE_REACQUIRED}]
    tentative_tracks = [track for track in all_tracks if track.state == STATE_TENTATIVE]
    elapsed_ms = round((time.perf_counter() - started) * 1000)
    return {
        "ok": True,
        "engine_version": "dynamic-watermark-v1",
        "brand_id": brand_id,
        "candidates": [item.model_dump() for item in candidates],
        "dynamic_tracks": [item.model_dump() for item in tracks],
        "tentative_tracks": [item.model_dump() for item in tentative_tracks],
        "fixed": [],
        "bottom": [],
        "uncertain": [item.model_dump() for item in candidates if _effective_score(item) < 0.80],
        "metrics": {
            "video_duration_seconds": duration,
            "sampled_frames": len(times),
            "detector_fps": float(detector_fps),
            "detector_runtime_ms": elapsed_ms,
            "icon_match_runtime_ms": elapsed_ms,
            "edge_match_enabled": templates.get("edge") is not None,
            "multi_scale_enabled": True,
            "ocr_roi_count": ocr_rois,
            "ocr_runtime_ms": 0,
            "tracker_runtime_ms": 0,
            "dense_tracker_generated_samples": dense_tracker_samples,
            "dense_tracker_policy": "continuous_only__kalman_prediction+optical_flow+local_template_refinement",
            "confirmed_track_count": len(tracks),
            "tentative_track_count": len(tentative_tracks),
            "cache_hit": False,
            "brand_profile_hit": True,
            "score_weights": {**DEFAULT_SCORE_WEIGHTS, **(weights or {})},
            "detector_policy": "visual_first__ocr_roi_confirmation_only",
            "tentative_policy": "strong_singleton_back_forward_visual_verification_0.5_to_1s",
            "scene_cut_policy": "hard_reset_screen_space_tracker",
        },
    }


class ScreenSpaceTrackBuilder:
    """Named façade for building DynamicWatermarkTrack[] without video I/O."""

    def __init__(self, *, max_gap: float = 1.25, max_reacquire_gap: float = 3.0):
        self.max_gap, self.max_reacquire_gap = float(max_gap), float(max_reacquire_gap)

    def build(self, candidates: Iterable[WatermarkCandidate], *, width: int = 1280, height: int = 720) -> list[DynamicWatermarkTrack]:
        return _track_candidates(list(candidates), max_gap=self.max_gap, frame_size=(int(width), int(height)), max_reacquire_gap=self.max_reacquire_gap)