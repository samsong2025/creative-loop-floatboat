"""Native worker bridge for halfzm/ProPainter-Webui.

The WebUI itself is interactive; production uses its
``ProPainter/inference_propainter.py`` entry point with a PNG mask per video
frame. Floatboat remains responsible for product-specific watermark identity
and trajectory verification. This module only rasterizes those verified tracks
and invokes ProPainter.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np


def _positive_int_env(name: str, default: int) -> int:
    """Read a positive integer deployment setting without breaking a job."""
    try:
        value = int(str(os.environ.get(name) or default).strip())
    except (TypeError, ValueError):
        value = default
    return max(1, value)


def _propainter_segment_frame_limit(
    backend: dict[str, Any],
    processing_profile: dict[str, Any],
    source_frame_count: int,
) -> int:
    """Return a bounded inference-window length for one ProPainter process.

    The upstream CLI loads every supplied frame into tensors before its own
    ``--subvideo_length`` batching takes effect. That setting therefore cannot
    cap peak memory for a merged trajectory window. On 8 GB GPUs, a 70+ frame
    768-pixel portrait window is enough to make the CUDA child exit by SIGKILL.

    Splitting protects memory only. The compositor still copies generated pixels
    only within verified masks and the existing receipt and residual-QA gates
    remain mandatory.
    """
    source_frame_count = max(1, int(source_frame_count or 1))
    raw = str(os.environ.get("PROPAINTER_MAX_SEGMENT_FRAMES") or "").strip()
    if raw:
        try:
            configured = int(raw)
        except (TypeError, ValueError):
            configured = 24
    else:
        configured = (
            24
            if processing_profile.get("profile_source") == "adaptive_low_vram"
            else source_frame_count
        )
    # Preserve enough temporal context while treating malformed configuration as
    # a safe bounded window rather than allowing an unbounded child process.
    return min(source_frame_count, max(12, min(120, configured)))


def probe_propainter_backend() -> dict[str, Any]:
    """Discover the configured video-repair backend.

    CUDA and ROCm both expose the PyTorch ``cuda`` API (ROCm additionally
    exposes ``torch.version.hip``).  DirectML is detected through
    ``torch_directml``.  When a ProPainter installation is unavailable we
    expose an explicit OpenCV temporal-mask worker so AMD/Intel/CPU hosts can
    still run the verified trajectory path instead of being hard-coded to an
    NVIDIA-only failure.
    """
    preference = str(os.environ.get("PROPAINTER_BACKEND") or "auto").strip().lower()
    if preference not in {"auto", "cuda", "rocm", "directml", "cpu", "opencv"}:
        preference = "auto"
    root_value = str(os.environ.get("PROPAINTER_ROOT") or "").strip()
    root = Path(root_value).expanduser() if root_value else None
    script = root / "ProPainter" / "inference_propainter.py" if root else None
    python_bin = str(os.environ.get("PROPAINTER_PYTHON") or "python")
    torch_info: dict[str, Any] = {}
    try:
        probe = subprocess.run(
            [python_bin, "-c", (
                "import torch; "
                "print(torch.version.cuda or ''); "
                "print(torch.version.hip or ''); "
                "print(int(torch.cuda.is_available())); "
                "print(int(torch.cuda.get_device_properties(0).total_memory "
                "// (1024 * 1024)) if torch.cuda.is_available() else 0)"
            )],
            capture_output=True, text=True, timeout=60, check=False,
        )
        # Keep positional empty fields: on a CUDA build ``torch.version.hip`` is
        # normally empty. Filtering blank lines would shift the final 0/1 CUDA
        # availability flag into the HIP field and falsely report no accelerator.
        lines = [line.strip() for line in probe.stdout.splitlines()]
        torch_info = {
            "torch_cuda_build": lines[0] if len(lines) > 0 and lines[0] else None,
            "torch_hip_build": lines[1] if len(lines) > 1 and lines[1] else None,
            "torch_accelerator_available": bool(len(lines) > 2 and lines[2] == "1"),
            "gpu_memory_total_mb": (
                int(lines[3])
                if len(lines) > 3 and lines[3].isdigit()
                else None
            ),
        }
    except Exception as exc:
        torch_info = {"probe_error": str(exc)}

    cuda_available = bool(torch_info.get("torch_accelerator_available"))
    hip_available = bool(torch_info.get("torch_hip_build")) and cuda_available
    directml_available = False
    try:
        dml_probe = subprocess.run(
            [python_bin, "-c", "import torch_directml; print(int(torch_directml.device_count() > 0))"],
            capture_output=True, text=True, timeout=30, check=False,
        )
        directml_available = dml_probe.stdout.strip().endswith("1")
    except Exception:
        directml_available = False

    accelerator = "rocm" if hip_available else ("cuda" if cuda_available else None)
    if preference == "rocm" and not hip_available:
        accelerator = None
    elif preference == "cuda" and not cuda_available:
        accelerator = None
    elif preference == "directml":
        accelerator = "directml" if directml_available else None

    if root and script and script.exists() and accelerator:
        return {
            "available": True,
            "status": f"propainter_{accelerator}_configured",
            "backend": accelerator,
            "root": str(root), "script": str(script), "python": python_bin,
            "transport": "subprocess", "supports_frame_masks": True,
            "directml_available": directml_available, **torch_info,
        }

    if preference in {"directml", "cpu", "opencv"} or (preference == "auto" and not (root and script and script.exists())):
        # Classical frame-mask repair is deliberately reported as a separate
        # backend. It is not mislabeled as generative ProPainter output.
        return {
            "available": True,
            "status": "opencv_temporal_mask_configured",
            "backend": "opencv",
            "accelerator": "directml" if preference == "directml" and directml_available else "cpu",
            "root": str(root) if root else None,
            "python": python_bin,
            "transport": "in_process",
            "supports_frame_masks": True,
            "directml_available": directml_available,
            **torch_info,
        }

    allow_cpu = str(os.environ.get("PROPAINTER_ALLOW_CPU", "1")).strip().lower() in {"1", "true", "yes", "on"}
    if allow_cpu:
        return {
            "available": True,
            "status": "opencv_temporal_mask_configured",
            "backend": "opencv",
            "accelerator": "cpu",
            "requested_backend": preference,
            "root": str(root) if root else None,
            "python": python_bin,
            "transport": "in_process",
            "supports_frame_masks": True,
            "fallback_reason": "requested accelerator unavailable; using CPU/OpenCV trajectory repair",
            "directml_available": directml_available,
            **torch_info,
        }
    return {
        "available": False,
        "status": "backend_unavailable",
        "backend": preference,
        "root": str(root) if root else None,
        "python": python_bin,
        "reason": "No compatible CUDA/ROCm ProPainter runtime is configured",
        "transport": "subprocess",
        "supports_frame_masks": True,
        "directml_available": directml_available,
        **torch_info,
    }


def _propainter_processing_profile(
    source_width: int,
    source_height: int,
    backend: dict[str, Any],
) -> dict[str, Any]:
    """Select a memory-safe ProPainter inference size.

    ProPainter restores the requested *processing* size in its output.  The
    caller subsequently rescales that candidate only inside verified masks,
    so this helper never changes the dimensions of the delivered video.  A
    full-resolution 1080x1920 portrait sequence needs substantially more than
    8 GB VRAM even in fp16 mode; defaulting such devices to a 768-pixel long
    edge avoids an OS-level SIGKILL while preserving the source aspect ratio.
    """
    if source_width <= 0 or source_height <= 0:
        return {
            "ok": False,
            "status": "invalid_source_dimensions",
            "source_dimensions": [source_width, source_height],
        }

    configured_width = str(os.environ.get("PROPAINTER_PROCESS_WIDTH") or "").strip()
    configured_height = str(os.environ.get("PROPAINTER_PROCESS_HEIGHT") or "").strip()
    if bool(configured_width) != bool(configured_height):
        return {
            "ok": False,
            "status": "incomplete_processing_dimensions",
            "message": (
                "PROPAINTER_PROCESS_WIDTH and PROPAINTER_PROCESS_HEIGHT must "
                "be set together so verified-mask geometry is not distorted."
            ),
        }

    profile_source = "native"
    if configured_width and configured_height:
        try:
            target_width = int(configured_width)
            target_height = int(configured_height)
        except ValueError:
            return {
                "ok": False,
                "status": "invalid_processing_dimensions",
                "configured_width": configured_width,
                "configured_height": configured_height,
            }
        if target_width <= 0 or target_height <= 0:
            return {
                "ok": False,
                "status": "invalid_processing_dimensions",
                "configured_width": configured_width,
                "configured_height": configured_height,
            }
        profile_source = "explicit_environment"
    else:
        try:
            gpu_memory_mb = int(backend.get("gpu_memory_total_mb") or 0)
        except (TypeError, ValueError):
            gpu_memory_mb = 0
        low_vram_limit_mb = _positive_int_env(
            "PROPAINTER_LOW_VRAM_MAX_MB", 10240
        )
        if gpu_memory_mb and gpu_memory_mb <= low_vram_limit_mb:
            long_edge = _positive_int_env("PROPAINTER_LOW_VRAM_LONG_EDGE", 768)
            scale = min(1.0, float(long_edge) / max(source_width, source_height))
            target_width = max(8, int(round(source_width * scale / 8.0)) * 8)
            target_height = max(8, int(round(source_height * scale / 8.0)) * 8)
            profile_source = "adaptive_low_vram"
        else:
            target_width, target_height = source_width, source_height

    # Resize masks and frames with the same aspect ratio.  The inference script
    # itself crops processing dimensions to multiples of eight, therefore doing
    # so here keeps the candidate dimensions predictable for verification.
    target_width = max(8, int(round(target_width / 8.0)) * 8)
    target_height = max(8, int(round(target_height / 8.0)) * 8)
    source_aspect = float(source_width) / float(source_height)
    target_aspect = float(target_width) / float(target_height)
    if abs(source_aspect - target_aspect) > 0.015:
        return {
            "ok": False,
            "status": "processing_aspect_ratio_mismatch",
            "source_dimensions": [source_width, source_height],
            "processing_dimensions": [target_width, target_height],
        }
    return {
        "ok": True,
        "source_dimensions": [source_width, source_height],
        "processing_dimensions": [target_width, target_height],
        "scaled": bool(
            target_width != source_width or target_height != source_height
        ),
        "profile_source": profile_source,
        "gpu_memory_total_mb": backend.get("gpu_memory_total_mb"),
    }


def _bbox_at(track: dict[str, Any], second: float) -> Optional[list[float]]:
    points = sorted(track.get("waypoints") or [], key=lambda p: float(p.get("t", 0.0)))
    if not points:
        return None
    window = track.get("visibility_window") or []
    gap = float(track.get("max_interpolation_gap_seconds") or 0.45)
    # Strict census tracks are source-of-truth observations.  They must not be
    # extrapolated before first sighting or after last sighting; allowing the
    # old compatibility hold here made the mask worker repair clean story
    # frames around every short watermark burst.
    strict_track = str(track.get("tracking_policy") or "").startswith((
        "verified_identity", "reviewed_template_identity"
    ))
    if len(window) == 2:
        start, end = float(window[0]), float(window[1])
        if strict_track and (second < start - 1e-6 or second > end + 1e-6):
            return None
        if not strict_track and (second < start - gap or second > end + gap):
            return None
    if second <= float(points[0].get("t", 0.0)):
        return list(points[0].get("bbox") or []) if second >= float(points[0].get("t", 0.0)) - gap else None
    if second >= float(points[-1].get("t", 0.0)):
        return list(points[-1].get("bbox") or []) if second <= float(points[-1].get("t", 0.0)) + gap else None
    for left, right in zip(points, points[1:]):
        t0, t1 = float(left.get("t", 0.0)), float(right.get("t", 0.0))
        if t0 <= second <= t1:
            a = 0.0 if t1 <= t0 else (second - t0) / (t1 - t0)
            b0, b1 = left.get("bbox") or [], right.get("bbox") or []
            if len(b0) != 4 or len(b1) != 4:
                return None
            return [float(x) + (float(y) - float(x)) * a for x, y in zip(b0, b1)]
    return None


def _template_glyph(template_path: Optional[str]) -> Optional[np.ndarray]:
    if not template_path:
        return None
    path = Path(template_path)
    if not path.exists():
        return None
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.size == 0:
        return None
    # A reviewed source-specific mask is already a binary glyph support image.
    # Use it as-is instead of applying a high-pass transform that would erase
    # its opaque interior and leave a visible wordmark body after repair.
    if "_mask_" in path.stem.lower() or "mask-" in path.stem.lower():
        return cv2.morphologyEx(
            np.where(image >= 32, 255, 0).astype(np.uint8),
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
    smooth = cv2.GaussianBlur(image, (0, 0), 7.0)
    signal = cv2.absdiff(image, smooth)
    # The reviewed wide wordmark is semi-transparent in the source. A strict
    # 15-level high-pass support catches only the brightest strokes and leaves
    # the broad alpha body visible. Use a conservative lower threshold, then
    # close/dilate only inside the tracked bbox so the mask follows the actual
    # wordmark without becoming a rectangle.
    default_threshold = 10 if image.shape[1] >= image.shape[0] * 2 else 15
    # The normal edge mask is conservative for unknown scenes. A reviewed
    # source-specific rerun can opt into a lower bounded threshold so the
    # translucent letter bodies are repaired without expanding beyond the
    # already-authorized tracking box.
    try:
        configured_threshold = int(
            str(os.environ.get("PROPAINTER_GLYPH_SIGNAL_THRESHOLD") or default_threshold)
        )
    except (TypeError, ValueError):
        configured_threshold = default_threshold
    threshold = max(2, min(64, configured_threshold))
    glyph = np.where(signal >= threshold, 255, 0).astype(np.uint8)
    return cv2.morphologyEx(glyph, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))


def _write_frame_masks(
    source_path: Path,
    mask_dir: Path,
    tracks: list[dict[str, Any]],
    *,
    template_path: Optional[str] = None,
    dilation: int = 2,
    mask_mode: str = "bbox",
    excluded_intervals: Optional[list[tuple[float, float]]] = None,
    time_offset_seconds: float = 0.0,
) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise RuntimeError("cannot open source video for ProPainter masks")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    mask_dir.mkdir(parents=True, exist_ok=True)
    glyph = _template_glyph(template_path)
    normalized_exclusions = [
        (float(start), float(end))
        for start, end in (excluded_intervals or [])
        if float(end) >= float(start)
    ]
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    written = 0
    track_mask_frames: dict[str, int] = {
        str(track.get("track_id") or track.get("cluster_id") or index): 0
        for index, track in enumerate(tracks, 1)
    }
    try:
        for index in range(max(0, count)):
            # Segment masks use local frame indices, while authoritative tracks
            # remain expressed on the source-video timeline.
            second = float(time_offset_seconds) + index / max(1e-6, fps)
            mask = np.zeros((height, width), dtype=np.uint8)
            if any(start <= second <= end for start, end in normalized_exclusions):
                cv2.imwrite(str(mask_dir / f"{index:06d}.png"), mask)
                written += 1
                continue
            for track_index, track in enumerate(tracks, 1):
                bbox = _strict_track_bbox_at(track, second)
                if not bbox or len(bbox) != 4:
                    continue
                x0 = max(0, min(width, int(round(bbox[0] * width))))
                y0 = max(0, min(height, int(round(bbox[1] * height))))
                x1 = max(0, min(width, int(round(bbox[2] * width))))
                y1 = max(0, min(height, int(round(bbox[3] * height))))
                if x1 <= x0 or y1 <= y0:
                    continue
                # The template's high-pass support catches only sharp letter
                # edges. Semi-transparent moving marks retain a broad alpha
                # body, so edge-only masks leave the watermark visible after
                # inpainting. A verified trajectory permits the complete
                # compact watermark bbox to be repaired; glyph mode remains
                # available for diagnosis.
                if mask_mode != "glyph" or glyph is None:
                    mask[y0:y1, x0:x1] = 255
                else:
                    local = cv2.resize(glyph, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
                    mask[y0:y1, x0:x1] = np.maximum(mask[y0:y1, x0:x1], local)
                track_key = str(track.get("track_id") or track.get("cluster_id") or track_index)
                track_mask_frames[track_key] = track_mask_frames.get(track_key, 0) + 1
            if dilation > 0:
                mask = cv2.dilate(mask, element, iterations=int(dilation))
            cv2.imwrite(str(mask_dir / f"{index:06d}.png"), mask)
            written += 1
    finally:
        cap.release()
    return {"fps": fps, "frame_count": count, "width": width, "height": height,
            "mask_frames_written": written, "mask_dir": str(mask_dir),
            "mask_mode": mask_mode,
            "processing_excluded_intervals": normalized_exclusions,
            "track_mask_frames": track_mask_frames}


def _read_frame_at_index(source_path: Path, frame_index: int) -> Optional[np.ndarray]:
    """Read one source frame without disturbing the sequential worker reader."""
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_index)))
        ok, frame = cap.read()
        return frame if ok and frame is not None else None
    finally:
        cap.release()


def _strict_track_bbox_at(track: dict[str, Any], second: float) -> Optional[list[float]]:
    """Return a mask bbox only inside a strict track's observed window."""
    window = track.get("visibility_window") or []
    if len(window) == 2 and not (float(window[0]) - 1e-6 <= second <= float(window[1]) + 1e-6):
        return None
    return _bbox_at(track, second)


def _inpaint_bbox_with_context(
    frame: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    *,
    local_mask: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Repair a verified bbox while retaining nearby scene pixels as context.

    Calling Telea on a crop whose complete extent is masked gives it no known
    pixels to synthesize from; OpenCV can then return the source untouched.
    Work on a padded crop and mask only the approved inner bbox so repair is
    still bounded to the trajectory but has a real surrounding context.
    """
    if frame is None or x1 <= x0 or y1 <= y0:
        return None
    frame_height, frame_width = frame.shape[:2]
    pad = max(6, min(24, int(round(min(x1 - x0, y1 - y0) * 0.18))))
    outer_x0, outer_y0 = max(0, x0 - pad), max(0, y0 - pad)
    outer_x1, outer_y1 = min(frame_width, x1 + pad), min(frame_height, y1 + pad)
    if outer_x1 <= outer_x0 or outer_y1 <= outer_y0:
        return None
    crop = frame[outer_y0:outer_y1, outer_x0:outer_x1].copy()
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    inner_y0, inner_y1 = y0 - outer_y0, y1 - outer_y0
    inner_x0, inner_x1 = x0 - outer_x0, x1 - outer_x0
    if local_mask is None or local_mask.shape[:2] != (y1 - y0, x1 - x0):
        mask[inner_y0:inner_y1, inner_x0:inner_x1] = 255
    else:
        mask[inner_y0:inner_y1, inner_x0:inner_x1] = np.where(
            local_mask >= 128, 255, 0
        ).astype(np.uint8)
    if not np.any(mask):
        return None
    repaired = cv2.inpaint(crop, mask, 3.0, cv2.INPAINT_TELEA)
    return repaired[inner_y0:inner_y1, inner_x0:inner_x1]


def _temporal_clean_patch(
    current: np.ndarray,
    reference: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> Optional[np.ndarray]:
    """Align a displaced clean ROI to the current scene before replacement.

    A rectangle-only Telea fill is not sufficient for a translucent moving
    icon: its broad alpha body survives as a dark/bright silhouette.  The
    reference is selected outside the track visibility window, then locally
    aligned so actor/camera motion does not turn the repair into a block.
    """
    if reference is None or current is None:
        return None
    if current.shape != reference.shape or x1 <= x0 or y1 <= y0:
        return None
    current_roi = current[y0:y1, x0:x1]
    reference_roi = reference[y0:y1, x0:x1]
    if current_roi.size == 0 or reference_roi.shape != current_roi.shape:
        return None
    try:
        current_gray = cv2.cvtColor(current_roi, cv2.COLOR_BGR2GRAY)
        reference_gray = cv2.cvtColor(reference_roi, cv2.COLOR_BGR2GRAY)

        # Farneback returns a displacement from the first image to the second.
        # The old implementation estimated current -> reference, then sampled
        # with that flow.  That is only safe when the flow is reliable; a
        # translucent watermark in the current ROI can make it lock onto the
        # watermark stroke and copy a hard rectangular scene mismatch.  Estimate
        # the clean-reference -> current displacement instead and validate the
        # warped plate against a narrow border before allowing it to replace
        # pixels.  An invalid plate deliberately returns None so the caller's
        # mask-bounded inpaint remains the safer fallback.
        flow = cv2.calcOpticalFlowFarneback(
            reference_gray, current_gray, None, 0.35, 3, 15, 3, 5, 1.1, 0
        )
        gx, gy = np.meshgrid(
            np.arange(reference_roi.shape[1], dtype=np.float32),
            np.arange(reference_roi.shape[0], dtype=np.float32),
        )
        aligned = cv2.remap(
            reference_roi,
            gx - flow[..., 0],
            gy - flow[..., 1],
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

        # Reject a reference that does not agree with the current scene around
        # the repair area.  The ring excludes the centre, where the competitor
        # mark is expected, and prevents a bad reference from becoming a visible
        # block over a face, garment, or background edge.
        height, width = current_gray.shape[:2]
        ring = np.zeros((height, width), dtype=np.uint8)
        margin = max(3, min(10, int(round(min(height, width) * 0.08))))
        ring[:margin, :] = 1
        ring[-margin:, :] = 1
        ring[:, :margin] = 1
        ring[:, -margin:] = 1
        current_border = current_gray[ring > 0].astype(np.float32)
        aligned_border = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)[ring > 0].astype(np.float32)
        if current_border.size == 0 or aligned_border.size == 0:
            return None
        border_error = float(np.median(np.abs(current_border - aligned_border)))
        border_scale = max(8.0, float(np.median(np.abs(current_border - np.median(current_border)))))
        if border_error > max(18.0, border_scale * 1.35):
            return None
        return aligned
    except Exception:
        return None


def _temporal_clean_patch_with_context(
    current: np.ndarray,
    reference: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> Optional[np.ndarray]:
    """Align a clean temporal plate using surrounding scene context.

    Flow estimated only inside a watermark box often locks onto the watermark
    itself. Estimate and validate on a padded neighbourhood, then crop the
    aligned plate back to the already-authorized tracked box.
    """
    if current is None or reference is None or current.shape != reference.shape:
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    frame_height, frame_width = current.shape[:2]
    pad = max(12, min(64, int(round(max(x1 - x0, y1 - y0) * 0.22))))
    outer_x0, outer_y0 = max(0, x0 - pad), max(0, y0 - pad)
    outer_x1, outer_y1 = min(frame_width, x1 + pad), min(frame_height, y1 + pad)
    if outer_x1 <= outer_x0 or outer_y1 <= outer_y0:
        return None
    current_outer = current[outer_y0:outer_y1, outer_x0:outer_x1]
    reference_outer = reference[outer_y0:outer_y1, outer_x0:outer_x1]
    try:
        current_gray = cv2.cvtColor(current_outer, cv2.COLOR_BGR2GRAY)
        reference_gray = cv2.cvtColor(reference_outer, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            reference_gray, current_gray, None, 0.35, 3, 21, 4, 7, 1.1, 0
        )
        gx, gy = np.meshgrid(
            np.arange(reference_outer.shape[1], dtype=np.float32),
            np.arange(reference_outer.shape[0], dtype=np.float32),
        )
        aligned_outer = cv2.remap(
            reference_outer, gx - flow[..., 0], gy - flow[..., 1],
            cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
        )
        ring = np.ones(current_gray.shape, dtype=np.uint8)
        inner_x0, inner_y0 = x0 - outer_x0, y0 - outer_y0
        inner_x1, inner_y1 = x1 - outer_x0, y1 - outer_y0
        ring[inner_y0:inner_y1, inner_x0:inner_x1] = 0
        if not np.any(ring):
            return None
        observed = current_gray[ring > 0].astype(np.float32)
        aligned = cv2.cvtColor(aligned_outer, cv2.COLOR_BGR2GRAY)[ring > 0].astype(np.float32)
        if observed.size == 0 or aligned.size == 0:
            return None
        border_error = float(np.median(np.abs(observed - aligned)))
        scene_scale = max(8.0, float(np.median(np.abs(observed - np.median(observed)))))
        if border_error > max(18.0, scene_scale * 1.35):
            return None
        return aligned_outer[inner_y0:inner_y1, inner_x0:inner_x1]
    except Exception:
        return None


def _inpaint_glyph_with_context(
    frame: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    glyph_mask: np.ndarray,
) -> Optional[np.ndarray]:
    """Inpaint only approved glyph pixels using a padded scene-context crop.

    A temporally sampled plate is useful on static backgrounds, but it proved
    visibly wrong when a source wordmark crosses a face or wardrobe.  This
    fallback keeps the mask's narrow glyph geometry while giving Telea enough
    same-frame context to reconstruct nearby texture; it never escalates to a
    full tracking rectangle.
    """
    if frame is None or x1 <= x0 or y1 <= y0 or glyph_mask is None:
        return None
    local_mask = np.where(glyph_mask >= 128, 255, 0).astype(np.uint8)
    if local_mask.shape[:2] != (y1 - y0, x1 - x0) or not np.any(local_mask):
        return None
    frame_height, frame_width = frame.shape[:2]
    pad = max(6, min(24, int(round(min(x1 - x0, y1 - y0) * 0.18))))
    outer_x0, outer_y0 = max(0, x0 - pad), max(0, y0 - pad)
    outer_x1, outer_y1 = min(frame_width, x1 + pad), min(frame_height, y1 + pad)
    if outer_x1 <= outer_x0 or outer_y1 <= outer_y0:
        return None
    crop = frame[outer_y0:outer_y1, outer_x0:outer_x1].copy()
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    inner_x0, inner_y0 = x0 - outer_x0, y0 - outer_y0
    inner_x1, inner_y1 = inner_x0 + (x1 - x0), inner_y0 + (y1 - y0)
    mask[inner_y0:inner_y1, inner_x0:inner_x1] = local_mask
    repaired = cv2.inpaint(crop, mask, 3.0, cv2.INPAINT_TELEA)
    return repaired[inner_y0:inner_y1, inner_x0:inner_x1]


def _run_opencv_temporal_mask_worker(
    source_path: Path,
    output_path: Path,
    mask_info: dict[str, Any],
) -> dict[str, Any]:
    """Portable non-CUDA repair backend for AMD, Intel and CPU hosts.

    This is intentionally a classical, explicitly-labelled backend: it uses
    the verified per-frame mask and OpenCV inpainting, never claims to be
    ProPainter/generative output, and remains subject to the same residual QA.
    """
    fps = float(mask_info.get("fps") or 30.0)
    width, height = int(mask_info.get("width") or 0), int(mask_info.get("height") or 0)
    if width <= 0 or height <= 0:
        return {"ok": False, "status": "invalid_video_metadata", "backend": "opencv"}
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        return {"ok": False, "status": "source_open_failed", "backend": "opencv"}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_video = output_path.with_suffix(".opencv-video.mp4")
    writer = cv2.VideoWriter(
        str(temp_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        cap.release()
        return {"ok": False, "status": "opencv_writer_unavailable", "backend": "opencv"}
    frames = 0
    tracks = list(mask_info.get("tracks") or [])
    mask_mode = str(mask_info.get("mask_mode") or "bbox").strip().lower()
    track_receipts: dict[str, int] = {
        str(track.get("track_id") or track.get("cluster_id") or index): 0
        for index, track in enumerate(tracks, 1)
    }
    changed_pixels_by_track: dict[str, int] = {
        key: 0 for key in track_receipts
    }
    # Build one clean reference per track before the sequential pass.  A frame
    # immediately outside a strict visibility window is not carrying that
    # tracked watermark at the tracked position, and is much safer than a
    # spatial-only inpaint for semi-transparent marks.
    reference_frames: dict[str, np.ndarray] = {}
    reference_frame_indices: dict[str, int] = {}
    frame_count = int(mask_info.get("frame_count") or 0)
    # References must be clean with respect to every approved dynamic track,
    # not merely the track currently being repaired. Back-to-back trajectories
    # otherwise let a preceding watermark burst become a false clean plate.
    occupied_frames: set[int] = set()
    for occupied_track in tracks:
        occupied_window = occupied_track.get("visibility_window") or []
        if len(occupied_window) != 2:
            continue
        occupied_start = max(0, int(math.floor(float(occupied_window[0]) * fps)))
        occupied_end = min(frame_count - 1, int(math.ceil(float(occupied_window[1]) * fps)))
        occupied_frames.update(range(occupied_start, occupied_end + 1))
    for index, track in enumerate(tracks, 1):
        key = str(track.get("track_id") or track.get("cluster_id") or index)
        window = track.get("visibility_window") or []
        if len(window) != 2:
            continue
        start_frame = max(0, int(round(float(window[0]) * fps)))
        end_frame = min(max(0, frame_count - 1), int(round(float(window[1]) * fps)))
        # A point immediately adjacent to a detector window is not a clean
        # reference for a translucent mark: the alpha fade can start before the
        # first positive template peak and end after the last one. Step beyond
        # the strict interpolation allowance before reading reference plates.
        # This prevents copying a still-visible watermark back into the repair.
        clearance = max(
            2,
            int(math.ceil(float(track.get("max_interpolation_gap_seconds") or 0.32) * fps)) + 2,
        )
        candidates = [
            start_frame - clearance,
            start_frame - clearance - 1,
            end_frame + clearance,
            end_frame + clearance + 1,
            start_frame - clearance * 2,
            end_frame + clearance * 2,
        ]
        for candidate_index in candidates:
            if (
                candidate_index < 0
                or candidate_index >= frame_count
                or candidate_index in occupied_frames
            ):
                continue
            reference = _read_frame_at_index(source_path, candidate_index)
            if reference is not None:
                reference_frames[key] = reference
                reference_frame_indices[key] = candidate_index
                break
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            mask_path = Path(mask_info["mask_dir"]) / f"{frames:06d}.png"
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None and np.any(mask):
                # Replace each tracked region from a locally aligned temporal
                # clean plate.  Keep Telea as a bounded fallback when no clean
                # reference can be read (for example at a clip boundary).
                second = frames / max(1e-6, fps)
                unresolved = mask.copy()
                # The reviewed v2 asset is a wide wordmark crop. Its glyph
                # support is the safe repair support; replacing the full
                # tracking rectangle copies scene motion (faces, garments, and
                # background edges) into a visible hard block. In glyph mode,
                # use an aligned temporal plate only on the reviewed wordmark
                # pixels, then inpaint any pixels whose reference failed.
                if mask_mode == "glyph":
                    for index, track in enumerate(tracks, 1):
                        key = str(track.get("track_id") or track.get("cluster_id") or index)
                        bbox = _strict_track_bbox_at(track, second)
                        reference = reference_frames.get(key)
                        if not bbox:
                            continue
                        x0 = max(0, min(width, int(round(bbox[0] * width))))
                        y0 = max(0, min(height, int(round(bbox[1] * height))))
                        x1 = max(0, min(width, int(round(bbox[2] * width))))
                        y1 = max(0, min(height, int(round(bbox[3] * height))))
                        if x1 <= x0 or y1 <= y0:
                            continue
                        local_mask = mask[y0:y1, x0:x1]
                        if local_mask.size == 0:
                            continue
                        # Prefer same-frame, glyph-bounded inpainting for the
                        # reviewed wordmark.  The source watermark frequently
                        # crosses moving faces and clothing, where a clean
                        # temporal plate creates a clearly visible displaced
                        # rectangle even if its edge metric looks improved.
                        patch = _inpaint_glyph_with_context(
                            frame, x0, y0, x1, y1, local_mask
                        )
                        if patch is None and reference is not None:
                            patch = _temporal_clean_patch(
                                frame, reference, x0, y0, x1, y1
                            )
                        if patch is None:
                            # Keep fallback attribution per authoritative track;
                            # a later whole-frame inpaint cannot prove which
                            # track was actually repaired.
                            local = frame[y0:y1, x0:x1]
                            before = local.copy()
                            fallback = cv2.inpaint(
                                local,
                                local_mask,
                                3.0,
                                cv2.INPAINT_TELEA,
                            )
                            local[:, :] = fallback
                            changed_pixels_by_track[key] += int(
                                np.count_nonzero(
                                    np.any(before != local, axis=2)
                                )
                            )
                            unresolved[y0:y1, x0:x1][local_mask >= 128] = 0
                            track_receipts[key] = track_receipts.get(key, 0) + 1
                            continue
                        core = local_mask >= 128
                        if not np.any(core):
                            continue
                        local = frame[y0:y1, x0:x1]
                        # Only the reviewed wordmark support is replaced. The
                        # surrounding subject/background remains from the
                        # current frame, preventing the previous rectangular
                        # ghost when the actor changes pose between references.
                        before = local[core].copy()
                        local[core] = patch[core]
                        changed_pixels_by_track[key] += int(
                            np.count_nonzero(np.any(before != local[core], axis=1))
                        )
                        edge_alpha = cv2.GaussianBlur(
                            local_mask, (0, 0), 0.9
                        ).astype(np.float32) / 255.0
                        edge = (edge_alpha > 0.0) & ~core
                        if np.any(edge):
                            blend = edge_alpha[..., None]
                            mixed = np.clip(
                                local.astype(np.float32) * (1.0 - blend)
                                + patch.astype(np.float32) * blend,
                                0,
                                255,
                            ).astype(np.uint8)
                            local[edge] = mixed[edge]
                        frame[y0:y1, x0:x1] = local
                        unresolved[y0:y1, x0:x1][core] = 0
                        track_receipts[key] = track_receipts.get(key, 0) + 1
                    if np.any(unresolved):
                        # Keep the fallback strictly inside the reviewed glyph
                        # mask. It must never become a broad ROI blur/fill.
                        frame = cv2.inpaint(frame, unresolved, 3.0, cv2.INPAINT_TELEA)
                    writer.write(frame)
                    frames += 1
                    continue
                for index, track in enumerate(tracks, 1):
                    key = str(track.get("track_id") or track.get("cluster_id") or index)
                    bbox = _strict_track_bbox_at(track, second)
                    reference = reference_frames.get(key)
                    if not bbox:
                        continue
                    x0 = max(0, min(width, int(round(bbox[0] * width))))
                    y0 = max(0, min(height, int(round(bbox[1] * height))))
                    x1 = max(0, min(width, int(round(bbox[2] * width))))
                    y1 = max(0, min(height, int(round(bbox[3] * height))))
                    if x1 <= x0 or y1 <= y0:
                        continue
                    local_mask = mask[y0:y1, x0:x1]
                    patch = (
                        _temporal_clean_patch_with_context(frame, reference, x0, y0, x1, y1)
                        if reference is not None else None
                    )
                    if local_mask.size == 0:
                        continue
                    if patch is None:
                        # Repair in a padded context crop. The changed-pixel
                        # receipt is attributed only to the verified bbox.
                        local = frame[y0:y1, x0:x1]
                        before = local.copy()
                        fallback = _inpaint_bbox_with_context(
                            frame, x0, y0, x1, y1, local_mask=local_mask
                        )
                        if fallback is None:
                            continue
                        local[:, :] = fallback
                        changed = int(np.count_nonzero(np.any(before != local, axis=2)))
                        changed_pixels_by_track[key] += changed
                        if changed <= 0:
                            unresolved[y0:y1, x0:x1][local_mask >= 192] = 0
                            continue
                        unresolved[y0:y1, x0:x1][local_mask >= 192] = 0
                        track_receipts[key] = track_receipts.get(key, 0) + 1
                        continue
                    core = local_mask >= 192
                    if not np.any(core):
                        continue
                    local = frame[y0:y1, x0:x1]
                    before = local[core].copy()
                    local[core] = patch[core]
                    changed = int(
                        np.count_nonzero(np.any(before != local[core], axis=1))
                    )
                    changed_pixels_by_track[key] += changed
                    if changed <= 0:
                        fallback = _inpaint_bbox_with_context(
                            frame, x0, y0, x1, y1, local_mask=local_mask
                        )
                        if fallback is not None:
                            local[:, :] = fallback
                            changed = int(np.count_nonzero(np.any(before != local[core], axis=1)))
                            changed_pixels_by_track[key] += changed
                    if changed <= 0:
                        unresolved[y0:y1, x0:x1][core] = 0
                        continue
                    edge_alpha = cv2.GaussianBlur(local_mask, (0, 0), 1.2).astype(np.float32) / 255.0
                    edge = (edge_alpha > 0.0) & ~core
                    if np.any(edge):
                        blend = edge_alpha[..., None]
                        mixed = np.clip(
                            local.astype(np.float32) * (1.0 - blend)
                            + patch.astype(np.float32) * blend,
                            0,
                            255,
                        ).astype(np.uint8)
                        local[edge] = mixed[edge]
                    frame[y0:y1, x0:x1] = local
                    unresolved[y0:y1, x0:x1][core] = 0
                    track_receipts[key] = track_receipts.get(key, 0) + 1
                # If a mask region had no usable reference, repair only that
                # remaining mask. Never expand the fallback beyond the mask.
                if np.any(unresolved):
                    frame = cv2.inpaint(frame, unresolved, 3.0, cv2.INPAINT_TELEA)
            writer.write(frame)
            frames += 1
    finally:
        cap.release()
        writer.release()
    if frames <= 0 or not temp_video.exists():
        return {"ok": False, "status": "opencv_no_frames", "backend": "opencv"}

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        # Some source creatives contain an audio stream longer than their video
        # stream. Apply the repaired video's measured frame duration as an
        # explicit mux limit; relying on ``-shortest`` with stream-copy audio
        # leaves long silent tails in those files.
        video_duration_seconds = frames / max(1e-6, fps)
        mux = subprocess.run(
            [ffmpeg, "-y", "-i", str(temp_video), "-i", str(source_path),
             "-map", "0:v:0", "-map", "1:a?", "-c:v", "libx264", "-preset", "veryfast",
             "-crf", "18", "-c:a", "aac", "-t", f"{video_duration_seconds:.6f}",
             "-shortest", str(output_path)],
            capture_output=True, text=True, timeout=600, check=False,
        )
        if mux.returncode != 0 or not output_path.exists():
            shutil.copyfile(temp_video, output_path)
    else:
        shutil.copyfile(temp_video, output_path)
    try:
        temp_video.unlink(missing_ok=True)
    except Exception:
        pass
    # Editorial replacement ranges are intentionally skipped when masks are
    # written. A strict track located entirely in one of those ranges cannot
    # receive a pixel receipt because none of its source frames can reach the
    # delivered timeline. Require receipts only for tracks that actually wrote
    # at least one retained-story mask frame; this preserves the hard receipt
    # contract for every repairable track without falsely rejecting the edit
    # because a later approved replacement removed another track wholesale.
    eligible_track_ids = [
        key for key, mask_frames in (mask_info.get("track_mask_frames") or {}).items()
        if int(mask_frames or 0) > 0
    ]
    all_tracks_applied = bool(eligible_track_ids) and all(
        int(track_receipts.get(key) or 0) > 0 for key in eligible_track_ids
    )
    all_tracks_changed = bool(eligible_track_ids) and all(
        int(changed_pixels_by_track.get(key) or 0) > 0 for key in eligible_track_ids
    )
    has_changed_pixels = any(
        int(changed_pixels_by_track.get(key) or 0) > 0 for key in eligible_track_ids
    )
    output_exists = output_path.exists() and output_path.stat().st_size > 1024
    return {
        # File existence alone is not a repair receipt.
        "ok": bool(output_exists and all_tracks_applied and all_tracks_changed and has_changed_pixels),
        "status": "completed" if output_exists and all_tracks_applied and all_tracks_changed and has_changed_pixels else "repair_not_applied",
        "backend": "opencv",
        "repair_mode": "glyph-mask-inpaint" if mask_mode == "glyph" else "temporal-clean-plate-mask",
        "frames": frames,
        "mask": mask_info,
        "track_receipts": track_receipts,
        "changed_pixels_by_track": changed_pixels_by_track,
        "receipt_kind": "applied_pixels",
        "reference_track_count": len(reference_frames),
        "reference_frame_indices": reference_frame_indices,
        "eligible_track_ids": eligible_track_ids,
        "tracks_with_mask_frames": [key for key, value in track_receipts.items() if value > 0],
    }


def run_propainter_frame_mask_worker(
    source_path: Path,
    output_path: Path,
    tracks: list[dict[str, Any]],
    *,
    metadata: Optional[dict[str, Any]] = None,
    timeout_seconds: Optional[float] = None,
    excluded_intervals: Optional[list[tuple[float, float]]] = None,
) -> dict[str, Any]:
    """Run GPU ProPainter only on bounded authoritative-track windows.

    A full source clip is never supplied to ProPainter: that causes its temporal
    model to retain unnecessary 1080x1920 frames and can be OOM-killed.  The
    repaired windows are composited back into a fresh full-length video, while
    per-track receipts compare source and generated pixels inside the exact
    approved masks.  A mask manifest alone is deliberately not a receipt.
    """
    backend = probe_propainter_backend()
    if not backend.get("available"):
        return {"ok": False, "status": "not_configured", "backend": backend}
    if backend.get("backend") == "opencv":
        # GPU-only deployments must not silently select the legacy worker.
        return {"ok": False, "status": "gpu_propainter_required", "backend": backend}
    source_path, output_path = Path(source_path), Path(output_path)
    if output_path.exists():
        return {"ok": False, "status": "output_path_already_exists", "backend": backend}
    root_value = backend.get("root")
    root = Path(root_value) if root_value else None
    if root is None:
        return {"ok": False, "status": "backend_root_missing", "backend": backend}
    timeout = float(timeout_seconds or os.environ.get("PROPAINTER_TIMEOUT_SECONDS") or 3600)
    supplied_metadata = metadata or {}
    selected_mask_mode = str(supplied_metadata.get("mask_mode") or os.environ.get("PROPAINTER_MASK_MODE") or "").strip().lower()
    template_value = str(supplied_metadata.get("template_path") or "")
    if not selected_mask_mode and ("_mask_" in Path(template_value).stem.lower() or "mask-" in Path(template_value).stem.lower()):
        selected_mask_mode = "glyph"
    if selected_mask_mode not in {"bbox", "glyph"}:
        selected_mask_mode = "bbox"
    try:
        selected_dilation = int(supplied_metadata.get("mask_dilation", os.environ.get("PROPAINTER_MASK_DILATION") or 2))
    except (TypeError, ValueError):
        selected_dilation = 2
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        return {"ok": False, "status": "source_open_failed", "backend": backend}
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    if frame_count <= 0 or width <= 0 or height <= 0:
        return {"ok": False, "status": "invalid_video_metadata", "backend": backend}
    processing_profile = _propainter_processing_profile(width, height, backend)
    if not processing_profile.get("ok"):
        return {
            "ok": False,
            "status": str(processing_profile.get("status") or "processing_profile_invalid"),
            "backend": backend,
            "processing_profile": processing_profile,
        }
    processing_width, processing_height = [
        int(value)
        for value in processing_profile["processing_dimensions"]
    ]
    segment_frame_limit = _propainter_segment_frame_limit(
        backend,
        processing_profile,
        frame_count,
    )
    try:
        context_frames = max(2, min(90, int(os.environ.get("PROPAINTER_SEGMENT_CONTEXT_FRAMES") or 18)))
    except ValueError:
        context_frames = 18
    windows: list[list[int]] = []
    for track in tracks:
        window = track.get("visibility_window") or []
        if len(window) != 2:
            continue
        start = max(0, int(math.floor(float(window[0]) * fps)) - context_frames)
        end = min(frame_count - 1, int(math.ceil(float(window[1]) * fps)) + context_frames)
        if end >= start:
            windows.append([start, end])
    windows.sort()
    merged_windows: list[list[int]] = []
    for start, end in windows:
        if merged_windows and start <= merged_windows[-1][1] + 1:
            merged_windows[-1][1] = max(merged_windows[-1][1], end)
        else:
            merged_windows.append([start, end])
    # ``--subvideo_length`` batches later inference stages only; the upstream
    # CLI still materializes every frame in this process first. Bound each child
    # process independently so nearby tracks cannot merge into an OOM-prone
    # 70+ frame portrait window on 8 GB GPUs.
    bounded_windows: list[list[int]] = []
    for start, end in merged_windows:
        cursor = int(start)
        while cursor <= int(end):
            bounded_end = min(int(end), cursor + segment_frame_limit - 1)
            # RAFT derives an optical-flow batch from consecutive frame pairs.
            # A final one-frame remainder therefore produces a zero-length batch
            # and fails at its correlation reshape. Overlap that tail with the
            # preceding segment instead: all original frames remain covered,
            # the compositor retains its source-timeline order, and every
            # ProPainter subprocess gets the minimum two-frame input it needs.
            if (
                bounded_end == int(end)
                and bounded_end == cursor
                and bounded_windows
                and int(bounded_windows[-1][1]) == cursor - 1
                and int(bounded_windows[-1][1]) - int(bounded_windows[-1][0]) + 1 > 2
            ):
                # Make the tail two frames long without overlap: move the last
                # frame of the preceding full segment into the one-frame tail.
                # Overlapping segments would make the sequential compositor
                # consume a candidate's frame N at source time N+1.
                bounded_windows[-1][1] = cursor - 2
                bounded_windows.append([cursor - 1, bounded_end])
            else:
                bounded_windows.append([cursor, bounded_end])
            cursor = bounded_end + 1
    merged_windows = bounded_windows
    if not merged_windows:
        return {"ok": False, "status": "no_authoritative_track_windows", "backend": backend}
    try:
        requested_subvideo_length = int(os.environ.get("PROPAINTER_SUBVIDEO_LENGTH") or 80)
    except ValueError:
        requested_subvideo_length = 80
    subvideo_length = max(3, min(80, requested_subvideo_length))
    track_receipts = {str(track.get("track_id") or track.get("cluster_id") or index): 0 for index, track in enumerate(tracks, 1)}
    changed_pixels_by_track = {key: 0 for key in track_receipts}
    track_output_frames = {key: 0 for key in track_receipts}
    segment_records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="creative-loop-propainter-segmented-") as temp:
        temp_dir = Path(temp)
        try:
            for segment_index, (start_frame, end_frame) in enumerate(merged_windows, 1):
                segment_dir = temp_dir / f"segment-{segment_index:03d}"
                segment_dir.mkdir(parents=True, exist_ok=True)
                segment_path = segment_dir / "source.mp4"
                writer = cv2.VideoWriter(str(segment_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                if not writer.isOpened():
                    return {"ok": False, "status": "segment_writer_unavailable", "backend": backend}
                reader = cv2.VideoCapture(str(source_path))
                try:
                    reader.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                    for _ in range(start_frame, end_frame + 1):
                        ok, frame = reader.read()
                        if not ok or frame is None:
                            return {"ok": False, "status": "segment_source_read_failed", "backend": backend}
                        writer.write(frame)
                finally:
                    reader.release()
                    writer.release()
                offset_seconds = start_frame / max(1e-6, fps)
                combined_mask_dir = segment_dir / "masks"
                combined_mask = _write_frame_masks(
                    segment_path, combined_mask_dir, tracks, template_path=template_value or None,
                    dilation=max(0, min(8, selected_dilation)), mask_mode=selected_mask_mode,
                    excluded_intervals=excluded_intervals, time_offset_seconds=offset_seconds,
                )
                per_track_mask_dirs: dict[str, Path] = {}
                per_track_mask_frames: dict[str, int] = {}
                for track_index, track in enumerate(tracks, 1):
                    key = str(track.get("track_id") or track.get("cluster_id") or track_index)
                    if int((combined_mask.get("track_mask_frames") or {}).get(key) or 0) <= 0:
                        continue
                    track_dir = segment_dir / f"track-mask-{track_index:03d}"
                    track_mask_info = _write_frame_masks(
                        segment_path, track_dir, [track], template_path=template_value or None,
                        dilation=max(0, min(8, selected_dilation)), mask_mode=selected_mask_mode,
                        excluded_intervals=excluded_intervals, time_offset_seconds=offset_seconds,
                    )
                    per_track_mask_dirs[key] = track_dir
                    per_track_mask_frames[key] = int(
                        (track_mask_info.get("track_mask_frames") or {}).get(key) or 0
                    )
                render_dir = segment_dir / "render"
                command = [os.environ.get("PROPAINTER_PYTHON") or "python", str(root / "ProPainter" / "inference_propainter.py"),
                           "--video", str(segment_path), "--mask", str(combined_mask_dir), "--output", str(render_dir),
                           "--subvideo_length", str(subvideo_length),
                           # The upstream script defaults to a four-pixel mask
                           # dilation, but the final source-size compositor
                           # previously copied only the undilated authoritative
                           # mask. Tell the worker to preserve exactly the same
                           # mask support it receives, so generated pixels are
                           # not discarded at the handoff boundary.
                           "--mask_dilation", "0"]
                # On lower-VRAM devices ProPainter runs at a bounded inference
                # size, then its candidate is restored to source dimensions
                # before generated pixels are copied only inside verified masks.
                # The delivered video therefore retains its original geometry.
                if processing_profile["scaled"]:
                    command.extend([
                        "--width", str(processing_width),
                        "--height", str(processing_height),
                    ])
                if str(os.environ.get("PROPAINTER_FP16") or "1").strip().lower() in {"1", "true", "yes", "on"}:
                    command.append("--fp16")
                proc = subprocess.run(command, cwd=str(root / "ProPainter"), capture_output=True, text=True, timeout=timeout, check=False)
                candidate = render_dir / segment_path.stem / "inpaint_out.mp4"
                if proc.returncode != 0 or not candidate.is_file() or candidate.stat().st_size <= 1024:
                    return {"ok": False, "status": "propainter_failed", "backend": backend, "return_code": proc.returncode,
                            "stderr": proc.stderr[-4000:], "stdout": proc.stdout[-2000:],
                            "reason": "segment_gpu_process_failed", "mask": combined_mask}
                candidate_cap = cv2.VideoCapture(str(candidate))
                try:
                    candidate_frames = int(candidate_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                    candidate_width = int(candidate_cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                    candidate_height = int(candidate_cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                finally:
                    candidate_cap.release()
                expected_segment_frames = end_frame - start_frame + 1
                if (
                    candidate_frames != expected_segment_frames
                    or candidate_width != processing_width
                    or candidate_height != processing_height
                ):
                    return {"ok": False, "status": "segment_output_metadata_mismatch", "backend": backend,
                            "expected_frames": expected_segment_frames,
                            "actual_frames": candidate_frames,
                            "expected_dimensions": [processing_width, processing_height],
                            "actual_dimensions": [candidate_width, candidate_height],
                            "mask": combined_mask,
                            "processing_profile": processing_profile}
                segment_records.append({
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "candidate": candidate,
                    "mask": combined_mask,
                    "per_track_mask_dirs": per_track_mask_dirs,
                    "per_track_mask_frames": per_track_mask_frames,
                    "output_frame_count": candidate_frames,
                })
            composite_video = temp_dir / "composited-video.mp4"
            writer = cv2.VideoWriter(str(composite_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            if not writer.isOpened():
                return {"ok": False, "status": "composite_writer_unavailable", "backend": backend}
            source_reader = cv2.VideoCapture(str(source_path))
            active_index = 0
            repair_reader: Optional[cv2.VideoCapture] = None
            try:
                for frame_index in range(frame_count):
                    ok, source_frame = source_reader.read()
                    if not ok or source_frame is None:
                        return {"ok": False, "status": "composite_source_read_failed", "backend": backend}
                    while active_index < len(segment_records) and frame_index > segment_records[active_index]["end_frame"]:
                        if repair_reader is not None:
                            repair_reader.release()
                            repair_reader = None
                        active_index += 1
                    output_frame = source_frame
                    if active_index < len(segment_records):
                        record = segment_records[active_index]
                        if record["start_frame"] <= frame_index <= record["end_frame"]:
                            if repair_reader is None:
                                repair_reader = cv2.VideoCapture(str(record["candidate"]))
                            repaired_ok, repaired_frame = repair_reader.read()
                            if not repaired_ok or repaired_frame is None:
                                return {"ok": False, "status": "segment_output_frame_mismatch", "backend": backend}
                            if repaired_frame.shape[:2] != source_frame.shape[:2]:
                                repaired_frame = cv2.resize(
                                    repaired_frame,
                                    (source_frame.shape[1], source_frame.shape[0]),
                                    interpolation=cv2.INTER_CUBIC,
                                )
                            if repaired_frame.shape != source_frame.shape:
                                return {
                                    "ok": False,
                                    "status": "segment_output_frame_mismatch",
                                    "backend": backend,
                                    "source_dimensions": [source_frame.shape[1], source_frame.shape[0]],
                                    "candidate_dimensions": [repaired_frame.shape[1], repaired_frame.shape[0]],
                                    "processing_profile": processing_profile,
                                }
                            # Do not replace a whole ProPainter segment: video
                            # decoding/encoding can alter clean pixels outside
                            # the approved support. Only copy generated pixels
                            # covered by the authoritative per-track masks.
                            output_frame = source_frame.copy()
                            local_index = frame_index - int(record["start_frame"])
                            for key, mask_dir in record["per_track_mask_dirs"].items():
                                track_mask = cv2.imread(
                                    str(mask_dir / f"{local_index:06d}.png"),
                                    cv2.IMREAD_GRAYSCALE,
                                )
                                if track_mask is None or not np.any(track_mask):
                                    continue
                                approved = track_mask >= 128
                                before = source_frame[approved]
                                generated = repaired_frame[approved]
                                changed = int(np.count_nonzero(np.any(before != generated, axis=1)))
                                output_frame[approved] = generated
                                track_output_frames[key] += 1
                                if changed > 0:
                                    track_receipts[key] += 1
                                    changed_pixels_by_track[key] += changed
                    writer.write(output_frame)
            finally:
                source_reader.release()
                writer.release()
                if repair_reader is not None:
                    repair_reader.release()
            ffmpeg = shutil.which("ffmpeg")
            if not ffmpeg:
                return {"ok": False, "status": "ffmpeg_required_for_audio_preservation", "backend": backend}
            output_path.parent.mkdir(parents=True, exist_ok=True)
            # Do not capture FFmpeg pipes: a long diagnostic stream can fill a
            # PIPE buffer and block the worker before it can finish the mux.
            # Preserve the source audio only for the actual repaired-video
            # duration. A malformed source may carry a much longer audio track;
            # copying that track without a hard ``-t`` creates a long silent
            # tail and makes the production compositor see the wrong duration.
            video_duration_seconds = frame_count / max(1e-6, fps)
            mux = subprocess.run(
                [ffmpeg, "-y", "-i", str(composite_video), "-i", str(source_path),
                 "-map", "0:v:0", "-map", "1:a?", "-c:v", "libx264",
                 "-preset", "veryfast", "-crf", "18", "-c:a", "aac",
                 "-t", f"{video_duration_seconds:.6f}", "-shortest",
                 str(output_path)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=600, check=False,
            )
            if mux.returncode != 0 or not output_path.is_file() or output_path.stat().st_size <= 1024:
                output_path.unlink(missing_ok=True)
                return {"ok": False, "status": "audio_mux_failed", "backend": backend,
                        "return_code": mux.returncode}
            expected_track_mask_frames = {
                key: sum(
                    int(record["per_track_mask_frames"].get(key) or 0)
                    for record in segment_records
                )
                for key in track_receipts
            }
            eligible_track_ids = [
                key for key, expected in expected_track_mask_frames.items()
                if expected > 0
            ]
            receipt_pass = bool(eligible_track_ids) and all(
                track_output_frames[key] == expected_track_mask_frames[key]
                and track_receipts[key] > 0
                and changed_pixels_by_track[key] > 0
                for key in eligible_track_ids
            )
            if not receipt_pass:
                # The caller must never discover a residual-failed partial file
                # at the requested output path and mistake it for an approved
                # repair artifact.
                output_path.unlink(missing_ok=True)
            return {"ok": bool(receipt_pass), "status": "completed" if receipt_pass else "repair_not_applied", "backend": backend,
                    "output_path": str(output_path) if receipt_pass else None,
                    "frames": frame_count, "track_count": len(tracks),
                    "mask": {"fps": fps, "frame_count": frame_count, "width": width, "height": height,
                             "mask_mode": selected_mask_mode,
                             "track_mask_frames": expected_track_mask_frames,
                             "output_covered_mask_frames": track_output_frames,
                             "segment_count": len(segment_records),
                             "segment_frame_limit": segment_frame_limit,
                             "processing_profile": processing_profile,
                             "segments": [
                                 {"start_frame": x["start_frame"], "end_frame": x["end_frame"],
                                  "output_frame_count": x["output_frame_count"]}
                                 for x in segment_records
                             ]},
                    "track_receipts": track_receipts, "changed_pixels_by_track": changed_pixels_by_track,
                    "receipt_kind": "generated_output_mask_pixel_delta", "eligible_track_ids": eligible_track_ids,
                    "segment_count": len(segment_records), "subvideo_length": subvideo_length,
                    "processing_profile": processing_profile}
        except subprocess.TimeoutExpired:
            output_path.unlink(missing_ok=True)
            return {"ok": False, "status": "propainter_timeout", "backend": backend}
        except Exception as exc:
            output_path.unlink(missing_ok=True)
            return {"ok": False, "status": "worker_exception", "backend": backend, "error": str(exc)}
