"""Native worker bridge for halfzm/ProPainter-Webui.

The WebUI itself is interactive; production uses its
``ProPainter/inference_propainter.py`` entry point with a PNG mask per video
frame. Floatboat remains responsible for product-specific watermark identity
and trajectory verification. This module only rasterizes those verified tracks
and invokes ProPainter.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np


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
                "print(int(torch.cuda.is_available()))"
            )],
            capture_output=True, text=True, timeout=60, check=False,
        )
        lines = [line.strip() for line in probe.stdout.splitlines() if line.strip()]
        torch_info = {
            "torch_cuda_build": lines[0] if lines else None,
            "torch_hip_build": lines[1] if len(lines) > 1 else None,
            "torch_accelerator_available": bool(len(lines) > 2 and lines[2] == "1"),
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


def _bbox_at(track: dict[str, Any], second: float) -> Optional[list[float]]:
    points = sorted(track.get("waypoints") or [], key=lambda p: float(p.get("t", 0.0)))
    if not points:
        return None
    window = track.get("visibility_window") or []
    gap = float(track.get("max_interpolation_gap_seconds") or 0.45)
    if len(window) == 2 and (second < float(window[0]) - gap or second > float(window[1]) + gap):
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
    smooth = cv2.GaussianBlur(image, (0, 0), 7.0)
    signal = cv2.absdiff(image, smooth)
    glyph = np.where(signal >= 15, 255, 0).astype(np.uint8)
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
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    written = 0
    try:
        for index in range(max(0, count)):
            second = index / max(1e-6, fps)
            mask = np.zeros((height, width), dtype=np.uint8)
            for track in tracks:
                bbox = _bbox_at(track, second)
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
            if dilation > 0:
                mask = cv2.dilate(mask, element, iterations=int(dilation))
            cv2.imwrite(str(mask_dir / f"{index:06d}.png"), mask)
            written += 1
    finally:
        cap.release()
    return {"fps": fps, "frame_count": count, "width": width, "height": height,
            "mask_frames_written": written, "mask_dir": str(mask_dir)}


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
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            mask_path = Path(mask_info["mask_dir"]) / f"{frames:06d}.png"
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None and np.any(mask):
                frame = cv2.inpaint(frame, mask, 3.0, cv2.INPAINT_TELEA)
            writer.write(frame)
            frames += 1
    finally:
        cap.release()
        writer.release()
    if frames <= 0 or not temp_video.exists():
        return {"ok": False, "status": "opencv_no_frames", "backend": "opencv"}

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        mux = subprocess.run(
            [ffmpeg, "-y", "-i", str(temp_video), "-i", str(source_path),
             "-map", "0:v:0", "-map", "1:a?", "-c:v", "libx264", "-preset", "veryfast",
             "-crf", "18", "-c:a", "copy", "-shortest", str(output_path)],
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
    return {"ok": output_path.exists() and output_path.stat().st_size > 1024,
            "status": "completed" if output_path.exists() else "encode_failed",
            "backend": "opencv", "frames": frames, "mask": mask_info}


def run_propainter_frame_mask_worker(
    source_path: Path,
    output_path: Path,
    tracks: list[dict[str, Any]],
    *,
    metadata: Optional[dict[str, Any]] = None,
    timeout_seconds: Optional[float] = None,
) -> dict[str, Any]:
    backend = probe_propainter_backend()
    if not backend.get("available"):
        return {"ok": False, "status": "not_configured", "backend": backend}
    source_path, output_path = Path(source_path), Path(output_path)
    timeout = float(timeout_seconds or os.environ.get("PROPAINTER_TIMEOUT_SECONDS") or 3600)
    root_value = backend.get("root")
    root = Path(root_value) if root_value else None
    with tempfile.TemporaryDirectory(prefix="creative-loop-propainter-") as temp:
        temp_dir = Path(temp)
        mask_dir = temp_dir / "masks"
        render_dir = temp_dir / "render"
        try:
            mask_info = _write_frame_masks(
                source_path, mask_dir, tracks,
                template_path=str((metadata or {}).get("template_path") or "") or None,
                dilation=int(os.environ.get("PROPAINTER_MASK_DILATION") or 2),
                mask_mode=str(os.environ.get("PROPAINTER_MASK_MODE") or "bbox").strip().lower(),
            )
            if backend.get("backend") == "opencv":
                result = _run_opencv_temporal_mask_worker(source_path, output_path, mask_info)
                result["backend"] = backend
                return result
            if root is None:
                return {"ok": False, "status": "backend_root_missing", "backend": backend, "mask": mask_info}
            script = root / "ProPainter" / "inference_propainter.py"
            command = [os.environ.get("PROPAINTER_PYTHON") or "python", str(script),
                       "--video", str(source_path), "--mask", str(mask_dir),
                       "--output", str(render_dir), "--subvideo_length",
                       str(int(os.environ.get("PROPAINTER_SUBVIDEO_LENGTH") or 80))]
            resize_ratio = float(os.environ.get("PROPAINTER_RESIZE_RATIO") or 1.0)
            if resize_ratio > 0 and resize_ratio != 1.0:
                command.extend(["--resize_ratio", str(resize_ratio)])
            if str(os.environ.get("PROPAINTER_FP16") or "1").strip().lower() in {"1", "true", "yes", "on"}:
                command.append("--fp16")
            proc = subprocess.run(command, cwd=str(root / "ProPainter"), capture_output=True,
                                  text=True, timeout=timeout, check=False)
            candidate = render_dir / source_path.stem / "inpaint_out.mp4"
            if proc.returncode != 0 or not candidate.exists():
                failure_reason = None
                if proc.returncode == 3221225477:
                    failure_reason = (
                        "ProPainter process terminated with Windows access-violation "
                        "(0xC0000005); check the selected accelerator runtime and memory"
                    )
                return {"ok": False, "status": "propainter_failed", "backend": backend,
                        "return_code": proc.returncode, "stderr": proc.stderr[-4000:],
                        "stdout": proc.stdout[-2000:], "reason": failure_reason,
                        "mask": mask_info}
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(candidate.read_bytes())
            if output_path.stat().st_size < 1024:
                return {"ok": False, "status": "invalid_output", "backend": backend, "mask": mask_info}
            return {"ok": True, "status": "completed", "backend": backend,
                    "output_path": str(output_path), "track_count": len(tracks), "mask": mask_info}
        except Exception as exc:
            return {"ok": False, "status": "worker_exception", "backend": backend, "error": str(exc)}
