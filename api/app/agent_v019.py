"""Creative Loop v0.19 Floatboat-facing control and AI packet API.

This module intentionally adapts the existing Operator Console task queue rather
than creating a second downloader, renderer or upload queue.  Floatboat gets
opaque task IDs and controlled artifact URLs; it never gets local file-system
paths, Insight credentials/cookies, or Mintegral credentials.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

try:
    from . import branding_v09 as _operator
except ImportError:  # pragma: no cover - permits direct module debugging
    import branding_v09 as _operator


AGENT_API_VERSION = "0.20.0"
_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_ALLOWED_ARTIFACT_SUFFIXES = {".jpg", ".jpeg", ".json"}
_IDEMPOTENCY_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$")
_IDEMPOTENCY_LOCK = threading.RLock()
_WORKBENCH_FILE = Path(__file__).resolve().parent / "static" / "creative_loop_workbench.html"


class AgentJobCreateRequest(BaseModel):
    """Public, review-first task contract used by the Floatboat workbench.

    The existing Operator queue remains authoritative.  This model deliberately
    exposes only task inputs an operator can see in the workbench and prevents
    callers from skipping the human review gate or enabling upload on creation.
    """

    idempotency_key: str = Field(min_length=1, max_length=160)
    mode: Literal["url", "search"]
    detail_urls: list[str] = Field(default_factory=list, max_length=50)
    keyword: str | None = Field(default=None, max_length=300)
    search_date_start: str | None = Field(default=None, max_length=10)
    search_date_end: str | None = Field(default=None, max_length=10)
    download_count: int = Field(default=10, ge=1, le=100)
    wait_for_review: Literal[True] = True
    auto_upload: Literal[False] = False


class AgentJobReviewRequest(BaseModel):
    """An explicit human decision for rendered items in a review-ready job."""

    decision: Literal["approve", "reject"]
    item_ids: list[str] = Field(default_factory=list, max_length=100)


class AgentJobCancelRequest(BaseModel):
    """Request cooperative cancellation at the next safe processing boundary."""

    reason: str = Field(default="", max_length=500)


class AgentBBox(BaseModel):
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class AgentFinding(BaseModel):
    kind: Literal["mid_promo", "dynamic_watermark", "end_card"]
    present: bool
    confidence: float = Field(ge=0.0, le=1.0)
    action: Literal["replace", "review", "ignore"] = "review"
    start_seconds: float | None = Field(default=None, ge=0.0)
    end_seconds: float | None = Field(default=None, ge=0.0)
    anchor_seconds: float | None = Field(default=None, ge=0.0)
    bbox: AgentBBox | None = None
    reason: str = Field(default="", max_length=2000)


class AgentDecisionRequest(BaseModel):
    packet_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
    packet_sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    provider: str = Field(min_length=1, max_length=80)
    model: str = Field(min_length=1, max_length=160)
    findings: list[AgentFinding] = Field(default_factory=list, max_length=40)
    summary: str = Field(default="", max_length=4000)


class AgentQCIssue(BaseModel):
    kind: Literal[
        "competitor_brand_residue",
        "dynamic_watermark_residue",
        "mid_promo_residue",
        "end_card_incorrect",
        "other",
    ]
    confidence: float = Field(ge=0.0, le=1.0)
    start_seconds: float | None = Field(default=None, ge=0.0)
    end_seconds: float | None = Field(default=None, ge=0.0)
    bbox: AgentBBox | None = None
    note: str = Field(default="", max_length=2000)


class AgentQCRequest(BaseModel):
    packet_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
    packet_sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    provider: str = Field(min_length=1, max_length=80)
    model: str = Field(min_length=1, max_length=160)
    outcome: Literal["passed", "failed", "needs_review"]
    issues: list[AgentQCIssue] = Field(default_factory=list, max_length=60)
    summary: str = Field(default="", max_length=4000)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _safe_segment(value: str, label: str) -> str:
    value = str(value or "").strip()
    if not _SAFE_SEGMENT.fullmatch(value):
        raise HTTPException(
            status_code=422,
            detail={"stage": "agent", "error": f"invalid_{label}"},
        )
    return value


def _workspace() -> Path:
    return Path(_operator.WORKSPACE).resolve()


def _agent_root(task_id: str) -> Path:
    safe_task_id = _safe_segment(task_id, "task_id")
    root = (_workspace() / "review" / "ai" / safe_task_id).resolve()
    review_root = (_workspace() / "review").resolve()
    if review_root not in root.parents:
        raise HTTPException(status_code=422, detail={"stage": "agent", "error": "path_escape"})
    root.mkdir(parents=True, exist_ok=True)
    return root


def _read_task(task_id: str) -> tuple[Path, dict[str, Any]]:
    safe_task_id = _safe_segment(task_id, "task_id")
    path, task = _operator._operator_read_job(safe_task_id)
    if not isinstance(task, dict):
        raise HTTPException(status_code=500, detail={"stage": "agent", "error": "task_state_invalid"})
    return path, task


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _task_item_summary(item: dict[str, Any]) -> dict[str, Any]:
    """Return safe-to-display media provenance; never return absolute paths."""
    item_id = str(item.get("item_id") or "")
    task_id = str(item.get("_task_id") or "")
    source_relative_path = str(item.get("source_video_relative_path") or "").replace("\\", "/").lstrip("/")
    processed_relative_path = str(item.get("processed_video_relative_path") or "").replace("\\", "/").lstrip("/")
    has_processed_output = bool(item.get("processed_video_relative_path"))
    return {
        "item_id": item_id,
        "status": str(item.get("status") or ""),
        "source_product": str(item.get("source_product_name") or item.get("product_name") or ""),
        "source_language": str(item.get("source_language") or item.get("language") or ""),
        "source_sha256": str(item.get("source_sha256") or ""),
        # These are workspace-relative provenance labels, not filesystem paths
        # accepted by any API. The workbench uses opaque IDs for media access.
        "source_reference": {
            "label": source_relative_path or None,
            "detail_url": str(item.get("source_detail_url") or item.get("detail_url") or "") or None,
        },
        "processed_reference": {
            "label": processed_relative_path or None,
            "metadata_label": str(item.get("processed_metadata_sidecar") or "").replace("\\", "/").lstrip("/") or None,
        },
        "business_qc_pass": bool((item.get("production_qc") or {}).get("branding_business_qc_pass")),
        "has_processed_output": has_processed_output,
        "preview_url": (
            f"/agent/v1/jobs/{task_id}/items/{item_id}/preview"
            if task_id and item_id and has_processed_output
            else None
        ),
    }


def _task_summary(task: dict[str, Any]) -> dict[str, Any]:
    request = task.get("request") if isinstance(task.get("request"), dict) else {}
    task_id = str(task.get("task_id") or "")
    items = []
    for item in task.get("items") or []:
        if isinstance(item, dict):
            item_with_context = dict(item)
            item_with_context["_task_id"] = task_id
            items.append(_task_item_summary(item_with_context))
    return {
        "ok": True,
        "api_version": AGENT_API_VERSION,
        "task_id": task_id,
        "job_id": task_id,
        "status": str(task.get("status") or ""),
        "stage_label": str(task.get("stage_label") or ""),
        "created_at": task.get("created_at"),
        "updated_at": task.get("updated_at"),
        "finished_at": task.get("finished_at"),
        "mode": str(request.get("mode") or ""),
        "source_request": {
            "detail_urls": [str(value) for value in request.get("detail_urls") or [] if isinstance(value, str) and value.strip()],
            "keyword": str(request.get("keyword") or "") or None,
            "search_date_start": str(request.get("search_date_start") or "") or None,
            "search_date_end": str(request.get("search_date_end") or "") or None,
        },
        "progress": task.get("progress") if isinstance(task.get("progress"), dict) else {},
        "summary": {
            **(task.get("summary") if isinstance(task.get("summary"), dict) else {}),
            "keyword": str(request.get("keyword") or ""),
            "detail_url_count": len(request.get("detail_urls") or []),
        },
        "items": items,
        "workbench_url": "/floatboat/workbench",
        "review_required": str(task.get("status") or "") == "AWAITING_REVIEW",
        "upload_requires_explicit_console_approval": True,
    }


def _safe_idempotency_key(value: str) -> str:
    value = str(value or "").strip()
    if not _IDEMPOTENCY_KEY.fullmatch(value):
        raise HTTPException(
            status_code=422,
            detail={"stage": "agent", "error": "invalid_idempotency_key"},
        )
    return value


def _idempotency_root() -> Path:
    root = (_workspace() / "state" / "agent_idempotency").resolve()
    state_root = (_workspace() / "state").resolve()
    if state_root not in root.parents:
        raise HTTPException(status_code=500, detail={"stage": "agent", "error": "idempotency_path_escape"})
    root.mkdir(parents=True, exist_ok=True)
    return root


def _idempotency_path(key: str) -> Path:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return _idempotency_root() / f"{digest}.json"


def _create_request_fingerprint(req: AgentJobCreateRequest) -> str:
    return _sha256_json(
        {
            "mode": req.mode,
            "detail_urls": [str(url).strip() for url in req.detail_urls],
            "keyword": str(req.keyword or "").strip(),
            "search_date_start": str(req.search_date_start or "").strip(),
            "search_date_end": str(req.search_date_end or "").strip(),
            "download_count": int(req.download_count),
            "wait_for_review": bool(req.wait_for_review),
            "auto_upload": bool(req.auto_upload),
        }
    )


def _create_job_sync(req: AgentJobCreateRequest) -> dict[str, Any]:
    """Create one Operator task, preserving idempotency across UI retries."""
    key = _safe_idempotency_key(req.idempotency_key)
    fingerprint = _create_request_fingerprint(req)
    record_path = _idempotency_path(key)

    with _IDEMPOTENCY_LOCK:
        if record_path.is_file():
            try:
                record = json.loads(record_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise HTTPException(status_code=500, detail={"stage": "agent", "error": "idempotency_record_unreadable"}) from exc
            if record.get("fingerprint") != fingerprint:
                raise HTTPException(status_code=409, detail={"stage": "agent", "error": "idempotency_key_reused_with_different_request"})
            task_id = _safe_segment(str(record.get("task_id") or ""), "task_id")
            _, task = _read_task(task_id)
            summary = _task_summary(task)
            summary["idempotent_replay"] = True
            return summary

        operator_req = _operator.OperatorTaskSubmitRequest(
            mode=req.mode,
            detail_urls=req.detail_urls if req.mode == "url" else None,
            keyword=req.keyword if req.mode == "search" else None,
            search_date_start=req.search_date_start if req.mode == "search" else None,
            search_date_end=req.search_date_end if req.mode == "search" else None,
            download_count=req.download_count,
        )
        created = _operator._operator_submit_sync(operator_req)
        task_id = _safe_segment(str(created.get("task_id") or ""), "task_id")
        _atomic_write_json(
            record_path,
            {
                "schema_version": 1,
                "created_at": _now(),
                "idempotency_key_sha256": hashlib.sha256(key.encode("utf-8")).hexdigest(),
                "fingerprint": fingerprint,
                "task_id": task_id,
            },
        )
        _, task = _read_task(task_id)
        summary = _task_summary(task)
        summary["idempotent_replay"] = False
        return summary


def _safe_processed_video(task_id: str, item_id: str) -> Path:
    """Resolve a processed review video from opaque IDs only.

    This is intentionally separate from the legacy Console media route: callers
    cannot supply a filesystem path and raw/intermediate media is never served.
    """
    safe_task_id = _safe_segment(task_id, "task_id")
    safe_item_id = _safe_segment(item_id, "item_id")
    _, task = _read_task(safe_task_id)
    item = next(
        (row for row in task.get("items") or [] if isinstance(row, dict) and str(row.get("item_id") or "") == safe_item_id),
        None,
    )
    if not item:
        raise HTTPException(status_code=404, detail={"stage": "agent", "error": "item_not_found"})
    relative = str(item.get("processed_video_relative_path") or "").replace("\\", "/").lstrip("/")
    if not relative.startswith("processed/") or Path(relative).suffix.lower() != ".mp4":
        raise HTTPException(status_code=404, detail={"stage": "agent", "error": "processed_preview_unavailable"})
    workspace = _workspace()
    path = (workspace / relative).resolve()
    processed_root = (workspace / "processed").resolve()
    if processed_root not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail={"stage": "agent", "error": "processed_preview_unavailable"})
    return path


def _video_stream_response(path: Path, request: Request) -> StreamingResponse:
    """Stream video with byte-range support for the Floatboat workbench."""
    file_size = path.stat().st_size
    range_header = request.headers.get("range")
    start, end, status_code = 0, file_size - 1, 200
    if range_header:
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
        if not match:
            raise HTTPException(status_code=416, detail="Invalid byte range.")
        raw_start, raw_end = match.groups()
        if raw_start:
            start = int(raw_start)
            end = int(raw_end) if raw_end else end
        elif raw_end:
            suffix = int(raw_end)
            start = max(0, file_size - suffix)
        if start >= file_size or start > end:
            raise HTTPException(status_code=416, detail="Requested byte range is not satisfiable.", headers={"Content-Range": f"bytes */{file_size}"})
        end = min(end, file_size - 1)
        status_code = 206
    content_length = end - start + 1

    def iter_file():
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = content_length
            while remaining > 0:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(content_length),
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0",
    }
    if status_code == 206:
        headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
    return StreamingResponse(iter_file(), status_code=status_code, media_type="video/mp4", headers=headers)


def _source_path_for_packet(item: dict[str, Any], packet_kind: str) -> tuple[Path, str]:
    relative = str(
        item.get("source_video_relative_path") if packet_kind == "analysis" else item.get("processed_video_relative_path")
        or ""
    ).replace("\\", "/").lstrip("/")
    required_prefix = "raw/" if packet_kind == "analysis" else "processed/"
    if not relative.startswith(required_prefix):
        raise ValueError(f"{packet_kind}_media_is_not_in_{required_prefix[:-1]}")
    path = (_workspace() / relative).resolve()
    if _workspace() not in path.parents or not path.is_file() or path.suffix.lower() != ".mp4":
        raise ValueError(f"{packet_kind}_media_unavailable")
    return path, relative


def _packet_revision(task: dict[str, Any], packet_kind: str, source_hashes: list[str]) -> str:
    return _sha256_json(
        {
            "packet_kind": packet_kind,
            "task_id": task.get("task_id"),
            "task_updated_at": task.get("updated_at"),
            "task_status": task.get("status"),
            "source_hashes": source_hashes,
            "packet_schema": 1,
        }
    )


def _read_video_samples(video_path: Path, positions: list[float]) -> tuple[dict[str, Any], list[tuple[float, np.ndarray]]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("video_open_failed")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = round(frame_count / fps, 3) if fps > 0 and frame_count > 0 else 0.0
        frames: list[tuple[float, np.ndarray]] = []
        for ratio in positions:
            seconds = max(0.0, min(duration, duration * ratio))
            capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000.0)
            ok, frame = capture.read()
            if ok and frame is not None and frame.size:
                frames.append((round(seconds, 3), frame))
        if not frames:
            raise ValueError("sample_frame_read_failed")
        return (
            {"duration_seconds": duration, "fps": round(fps, 3), "frame_count": frame_count, "width": width, "height": height},
            frames,
        )
    finally:
        capture.release()


def _write_contact_sheet(samples: list[tuple[float, np.ndarray]], output_path: Path) -> None:
    tile_w, tile_h, label_h = 320, 180, 28
    columns = 3
    rows = int(np.ceil(len(samples) / columns))
    sheet = np.full((rows * (tile_h + label_h), columns * tile_w, 3), 245, dtype=np.uint8)
    for index, (seconds, frame) in enumerate(samples):
        resized = cv2.resize(frame, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        row, column = divmod(index, columns)
        y, x = row * (tile_h + label_h), column * tile_w
        sheet[y : y + tile_h, x : x + tile_w] = resized
        cv2.putText(
            sheet,
            f"{seconds:.2f}s",
            (x + 9, y + tile_h + 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (35, 42, 50),
            1,
            cv2.LINE_AA,
        )
    if not cv2.imwrite(str(output_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 88]):
        raise ValueError("contact_sheet_write_failed")


def _artifact_url(task_id: str, packet_id: str, filename: str) -> str:
    return f"/agent/jobs/{task_id}/artifacts/{packet_id}/{filename}"


def _load_manifest(task_id: str, packet_id: str) -> tuple[Path, dict[str, Any]]:
    packet_dir = _agent_root(task_id) / _safe_segment(packet_id, "packet_id")
    manifest_path = packet_dir / "manifest.json"
    if not manifest_path.is_file():
        raise HTTPException(status_code=404, detail={"stage": "agent", "error": "packet_not_found"})
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"stage": "agent", "error": "packet_unreadable", "message": str(exc)[:300]}) from exc
    if not isinstance(manifest, dict):
        raise HTTPException(status_code=500, detail={"stage": "agent", "error": "packet_invalid"})
    return packet_dir, manifest


def _validate_packet_binding(task_id: str, packet_id: str, packet_sha256: str) -> tuple[Path, dict[str, Any]]:
    packet_dir, manifest = _load_manifest(task_id, packet_id)
    actual = str(manifest.get("packet_sha256") or "").lower()
    if not actual or actual != str(packet_sha256).lower():
        raise HTTPException(status_code=409, detail={"stage": "agent", "error": "packet_sha256_mismatch"})
    return packet_dir, manifest


def _validate_span(start: float | None, end: float | None, duration: float) -> None:
    if start is None and end is None:
        return
    if start is None or end is None or start >= end or end > duration + 0.05:
        raise HTTPException(status_code=422, detail={"stage": "agent", "error": "invalid_time_span", "duration_seconds": duration})


def _validate_time_against_packet(start: float | None, end: float | None, items: list[dict[str, Any]]) -> None:
    if start is None and end is None:
        return
    if not items:
        raise HTTPException(status_code=409, detail={"stage": "agent", "error": "packet_has_no_items"})
    max_duration = max(float((item.get("video") or {}).get("duration_seconds") or 0.0) for item in items)
    _validate_span(start, end, max_duration)


def _validate_anchor_against_packet(anchor: float | None, items: list[dict[str, Any]]) -> None:
    if anchor is None:
        return
    if not items:
        raise HTTPException(status_code=409, detail={"stage": "agent", "error": "packet_has_no_items"})
    max_duration = max(float((item.get("video") or {}).get("duration_seconds") or 0.0) for item in items)
    if anchor > max_duration + 0.05:
        raise HTTPException(status_code=422, detail={"stage": "agent", "error": "anchor_out_of_range", "duration_seconds": max_duration})


def _validate_bbox_against_packet(bbox: AgentBBox | None, items: list[dict[str, Any]]) -> None:
    if bbox is None:
        return
    if not items:
        raise HTTPException(status_code=409, detail={"stage": "agent", "error": "packet_has_no_items"})
    matching_dimensions = [
        item.get("video") or {}
        for item in items
        if bbox.x + bbox.width <= int((item.get("video") or {}).get("width") or 0)
        and bbox.y + bbox.height <= int((item.get("video") or {}).get("height") or 0)
    ]
    if not matching_dimensions:
        raise HTTPException(status_code=422, detail={"stage": "agent", "error": "bbox_out_of_bounds"})


def _validate_bbox(bbox: AgentBBox | None, width: int, height: int) -> None:
    if bbox is None:
        return
    if bbox.x + bbox.width > width or bbox.y + bbox.height > height:
        raise HTTPException(status_code=422, detail={"stage": "agent", "error": "bbox_out_of_bounds"})


def _write_append_only_record(packet_dir: Path, prefix: str, payload: dict[str, Any]) -> str:
    digest = _sha256_json(payload)[:16]
    name = f"{prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{digest}.json"
    _atomic_write_json(packet_dir / name, payload)
    return name


def _prepare_packet_sync(task_id: str, packet_kind: Literal["analysis", "qc"]) -> dict[str, Any]:
    _, task = _read_task(task_id)
    candidates: list[tuple[dict[str, Any], Path, str, str]] = []
    for item in task.get("items") or []:
        if not isinstance(item, dict):
            continue
        try:
            source_path, _ = _source_path_for_packet(item, packet_kind)
            source_hash = _sha256_file(source_path)
        except ValueError:
            continue
        candidates.append((item, source_path, source_hash, str(item.get("item_id") or "")))

    if not candidates:
        required = "processed outputs that passed local production QC" if packet_kind == "qc" else "archived raw videos"
        raise HTTPException(status_code=409, detail={"stage": "agent_packet", "error": "no_eligible_media", "required": required})

    revision = _packet_revision(task, packet_kind, [row[2] for row in candidates])
    packet_id = f"{packet_kind}-{revision[:16]}"
    packet_dir = _agent_root(task_id) / packet_id
    manifest_path = packet_dir / "manifest.json"
    if manifest_path.is_file():
        _, existing = _load_manifest(task_id, packet_id)
        return existing

    packet_dir.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, Any]] = []
    for item, video_path, source_hash, item_id in candidates:
        positions = [0.04, 0.20, 0.38, 0.56, 0.74, 0.92] if packet_kind == "analysis" else [0.05, 0.25, 0.50, 0.75, 0.95]
        video, samples = _read_video_samples(video_path, positions)
        item_key = _safe_segment(item_id or f"item-{len(items) + 1}", "item_id")
        image_records = []
        for index, (seconds, frame) in enumerate(samples, start=1):
            filename = f"{item_key}-sample-{index:02d}.jpg"
            path = packet_dir / filename
            if not cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 87]):
                raise HTTPException(status_code=500, detail={"stage": "agent_packet", "error": "sample_write_failed"})
            image_records.append(
                {
                    "timestamp_seconds": seconds,
                    "sha256": _sha256_file(path),
                    "url": _artifact_url(task_id, packet_id, filename),
                }
            )
        sheet_name = f"{item_key}-contact-sheet.jpg"
        _write_contact_sheet(samples, packet_dir / sheet_name)
        items.append(
            {
                "item_id": item_id,
                "source_sha256": source_hash,
                "source_product": str(item.get("source_product_name") or item.get("product_name") or ""),
                "source_language": str(item.get("source_language") or item.get("language") or ""),
                "video": video,
                "contact_sheet": {
                    "sha256": _sha256_file(packet_dir / sheet_name),
                    "url": _artifact_url(task_id, packet_id, sheet_name),
                },
                "samples": image_records,
                "local_business_qc": (item.get("production_qc") or {}).get("branding_business_qc") if packet_kind == "qc" else None,
            }
        )

    bindable = {
        "schema_version": 1,
        "api_version": AGENT_API_VERSION,
        "task_id": task_id,
        "task_status": str(task.get("status") or ""),
        "task_updated_at": task.get("updated_at"),
        "packet_id": packet_id,
        "packet_kind": packet_kind,
        "items": items,
        "privacy": {
            "contains_full_video": False,
            "contains_local_paths": False,
            "contains_browser_credentials": False,
            "contains_mtg_credentials": False,
        },
    }
    manifest = dict(bindable)
    manifest["packet_sha256"] = _sha256_json(bindable)
    _atomic_write_json(manifest_path, manifest)
    return manifest


def register_agent_routes(app) -> None:
    @app.get("/floatboat/workbench", include_in_schema=False)
    async def creative_loop_floatboat_workbench():
        """Serve the shared workbench for Floatboat Preview and the browser."""
        if not _WORKBENCH_FILE.is_file():
            raise HTTPException(status_code=500, detail={"stage": "agent", "error": "workbench_asset_missing"})
        return HTMLResponse(_WORKBENCH_FILE.read_text(encoding="utf-8"))

    @app.get("/agent/health", tags=["agent"])
    async def creative_loop_agent_health():
        state = _operator._operator_system_state_sync()
        return {
            "ok": True,
            "api_version": AGENT_API_VERSION,
            "service": "creative-loop-agent-api",
            "console_url": "http://localhost:8000/console",
            "operator_ready": bool(state.get("ok")),
            "insight_logged_in": bool((state.get("login") or {}).get("logged_in")),
            "brand_assets_ready": bool((state.get("brand_assets") or {}).get("ready")),
            "mintegral_ready": bool((state.get("mintegral") or {}).get("ready")),
            "security_note": "This endpoint never returns local paths, browser cookies/tokens, Insight credentials, or Mintegral credentials.",
        }

    @app.get("/agent/v1/capabilities", tags=["agent-v1"])
    async def creative_loop_agent_capabilities():
        return {
            "ok": True,
            "api_version": AGENT_API_VERSION,
            "workbench_url": "/floatboat/workbench",
            "capabilities": {
                "search_job": True,
                "detail_url_job": True,
                "job_idempotency": True,
                "job_cancel": True,
                "review_before_upload": True,
                "processed_video_preview": True,
                "insight_login_launch": True,
                "automatic_upload_on_create": False,
            },
            "limits": {"max_detail_urls": 50, "max_download_count": 100, "max_list_limit": 100},
            "security_note": "The Engine accepts no local paths or credentials and never allows automatic upload when a job is created.",
        }

    @app.get("/agent/v1/health", tags=["agent-v1"])
    async def creative_loop_agent_v1_health():
        return await creative_loop_agent_health()

    @app.post("/agent/v1/login/open", tags=["agent-v1"])
    async def creative_loop_agent_login_open():
        result = await asyncio.to_thread(_operator._operator_login_open_sync)
        return {
            "ok": bool(result.get("ok")),
            "api_version": AGENT_API_VERSION,
            "starting": bool(result.get("starting")),
            "already_running": bool(result.get("already_running")),
            "vnc_login_url": result.get("vnc_url"),
            "message": str(result.get("message") or ""),
        }

    @app.post("/agent/v1/jobs", tags=["agent-v1"])
    async def creative_loop_agent_create_job(req: AgentJobCreateRequest):
        return await asyncio.to_thread(_create_job_sync, req)

    @app.get("/agent/v1/jobs", tags=["agent-v1"])
    async def creative_loop_agent_v1_jobs(limit: int = 20):
        if not 1 <= limit <= 100:
            raise HTTPException(status_code=422, detail={"stage": "agent", "error": "limit_out_of_range"})
        rows = await asyncio.to_thread(_operator._operator_tasks_sync, limit)
        return {
            "ok": True,
            "api_version": AGENT_API_VERSION,
            "count": int(rows.get("count") or 0),
            "items": [_task_summary(task) for task in rows.get("items") or [] if isinstance(task, dict)],
        }

    @app.get("/agent/v1/jobs/{task_id}", tags=["agent-v1"])
    async def creative_loop_agent_v1_job(task_id: str):
        _, task = await asyncio.to_thread(_read_task, task_id)
        return _task_summary(task)

    @app.post("/agent/v1/jobs/{task_id}/review", tags=["agent-v1"])
    async def creative_loop_agent_review(task_id: str, req: AgentJobReviewRequest):
        safe_task_id = _safe_segment(task_id, "task_id")
        result = await asyncio.to_thread(
            _operator._operator_review_sync,
            _operator.OperatorReviewRequest(task_id=safe_task_id, decision=req.decision, item_ids=req.item_ids),
        )
        return {
            "ok": bool(result.get("ok")),
            "api_version": AGENT_API_VERSION,
            "task_id": safe_task_id,
            "job_id": safe_task_id,
            "decision": req.decision,
            "status": "review_accepted",
            "message": str(result.get("message") or ""),
            "upload_started": req.decision == "approve",
        }

    @app.post("/agent/v1/jobs/{task_id}/cancel", tags=["agent-v1"])
    async def creative_loop_agent_cancel(task_id: str, req: AgentJobCancelRequest):
        safe_task_id = _safe_segment(task_id, "task_id")
        result = await asyncio.to_thread(
            _operator._operator_cancel_sync,
            _operator.OperatorCancelRequest(task_id=safe_task_id),
        )
        return {
            "ok": bool(result.get("ok")),
            "api_version": AGENT_API_VERSION,
            "task_id": safe_task_id,
            "job_id": safe_task_id,
            "status": str(result.get("status") or "CANCEL_REQUESTED"),
            "safe_boundary_wait": bool(result.get("safe_boundary_wait")),
            "message": str(result.get("message") or ""),
        }

    @app.get("/agent/v1/jobs/{task_id}/items/{item_id}/preview", tags=["agent-v1"])
    async def creative_loop_agent_processed_preview(task_id: str, item_id: str, request: Request):
        path = await asyncio.to_thread(_safe_processed_video, task_id, item_id)
        return _video_stream_response(path, request)

    @app.get("/agent/console/session", tags=["agent"])
    async def creative_loop_agent_console_session():
        state = _operator._operator_system_state_sync()
        return {
            "ok": True,
            "api_version": AGENT_API_VERSION,
            "console_url": "http://localhost:8000/console",
            "vnc_login_url": "http://localhost:7900/?autoconnect=1&resize=scale",
            "insight_logged_in": bool((state.get("login") or {}).get("logged_in")),
            "brand_assets_ready": bool((state.get("brand_assets") or {}).get("ready")),
            "mintegral_ready": bool((state.get("mintegral") or {}).get("ready")),
            "next_user_step": "Enter either one Insight detail URL or a keyword/date range in the Console, then review the rendered output before approving MTG upload.",
        }

    @app.get("/agent/jobs/{task_id}", tags=["agent"])
    async def creative_loop_agent_job(task_id: str):
        _, task = await asyncio.to_thread(_read_task, task_id)
        return _task_summary(task)

    @app.get("/agent/jobs", tags=["agent"])
    async def creative_loop_agent_jobs(limit: int = 20):
        rows = _operator._operator_tasks_sync(limit=limit)
        return {
            "ok": True,
            "api_version": AGENT_API_VERSION,
            "count": int(rows.get("count") or 0),
            "items": [_task_summary(task) for task in rows.get("items") or [] if isinstance(task, dict)],
        }

    @app.post("/agent/jobs/{task_id}/ai-packet", tags=["agent"])
    async def creative_loop_agent_analysis_packet(task_id: str):
        return await __import__("asyncio").to_thread(_prepare_packet_sync, task_id, "analysis")

    @app.post("/agent/jobs/{task_id}/qc-packet", tags=["agent"])
    async def creative_loop_agent_qc_packet(task_id: str):
        return await __import__("asyncio").to_thread(_prepare_packet_sync, task_id, "qc")

    @app.get("/agent/jobs/{task_id}/artifacts/{packet_id}/{artifact_name}", tags=["agent"])
    async def creative_loop_agent_artifact(task_id: str, packet_id: str, artifact_name: str):
        _safe_segment(task_id, "task_id")
        _safe_segment(packet_id, "packet_id")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,180}", str(artifact_name or "")):
            raise HTTPException(status_code=422, detail={"stage": "agent", "error": "invalid_artifact_name"})
        packet_dir, _ = _load_manifest(task_id, packet_id)
        artifact = (packet_dir / artifact_name).resolve()
        if packet_dir.resolve() not in artifact.parents or not artifact.is_file() or artifact.suffix.lower() not in _ALLOWED_ARTIFACT_SUFFIXES:
            raise HTTPException(status_code=404, detail={"stage": "agent", "error": "artifact_not_found"})
        media_type = "image/jpeg" if artifact.suffix.lower() in {".jpg", ".jpeg"} else "application/json"
        return FileResponse(str(artifact), media_type=media_type, filename=artifact.name)

    @app.post("/agent/jobs/{task_id}/ai-decision", tags=["agent"])
    async def creative_loop_agent_decision(task_id: str, req: AgentDecisionRequest):
        _, task = _read_task(task_id)
        packet_dir, manifest = _validate_packet_binding(task_id, req.packet_id, req.packet_sha256)
        if manifest.get("packet_kind") != "analysis":
            raise HTTPException(status_code=409, detail={"stage": "agent", "error": "analysis_packet_required"})
        packet_items = [item for item in manifest.get("items") or [] if isinstance(item, dict)]
        for finding in req.findings:
            _validate_time_against_packet(finding.start_seconds, finding.end_seconds, packet_items)
            _validate_anchor_against_packet(finding.anchor_seconds, packet_items)
            _validate_bbox_against_packet(finding.bbox, packet_items)
        record = {
            "schema_version": 1,
            "record_type": "floatboat_ai_decision",
            "recorded_at": _now(),
            "task_id": task_id,
            "task_updated_at": task.get("updated_at"),
            "packet_id": req.packet_id,
            "packet_sha256": req.packet_sha256.lower(),
            "provider": req.provider,
            "model": req.model,
            "findings": [finding.model_dump() for finding in req.findings],
            "summary": req.summary,
            "applied_to_replacement_plan": False,
            "next_step": "Console/operator review required before any plan mutation; deterministic renderer remains authoritative.",
        }
        record_name = _write_append_only_record(packet_dir, "floatboat-decision", record)
        return {
            "ok": True,
            "api_version": AGENT_API_VERSION,
            "task_id": task_id,
            "status": "decision_recorded",
            "record_url": _artifact_url(task_id, req.packet_id, record_name),
            "applied_to_replacement_plan": False,
        }

    @app.post("/agent/jobs/{task_id}/ai-qc", tags=["agent"])
    async def creative_loop_agent_qc(task_id: str, req: AgentQCRequest):
        _, task = _read_task(task_id)
        packet_dir, manifest = _validate_packet_binding(task_id, req.packet_id, req.packet_sha256)
        if manifest.get("packet_kind") != "qc":
            raise HTTPException(status_code=409, detail={"stage": "agent", "error": "qc_packet_required"})
        packet_items = [item for item in manifest.get("items") or [] if isinstance(item, dict)]
        for issue in req.issues:
            _validate_time_against_packet(issue.start_seconds, issue.end_seconds, packet_items)
            _validate_bbox_against_packet(issue.bbox, packet_items)
        record = {
            "schema_version": 1,
            "record_type": "floatboat_ai_qc",
            "recorded_at": _now(),
            "task_id": task_id,
            "task_updated_at": task.get("updated_at"),
            "packet_id": req.packet_id,
            "packet_sha256": req.packet_sha256.lower(),
            "provider": req.provider,
            "model": req.model,
            "outcome": req.outcome,
            "issues": [issue.model_dump() for issue in req.issues],
            "summary": req.summary,
            "automatic_repair_started": False,
            "next_step": "Use Console review for approval/upload. Failed or uncertain AI QC remains an auditable review signal until local repair routing is enabled.",
        }
        record_name = _write_append_only_record(packet_dir, "floatboat-qc", record)
        return {
            "ok": True,
            "api_version": AGENT_API_VERSION,
            "task_id": task_id,
            "status": "qc_recorded",
            "outcome": req.outcome,
            "record_url": _artifact_url(task_id, req.packet_id, record_name),
            "automatic_repair_started": False,
        }