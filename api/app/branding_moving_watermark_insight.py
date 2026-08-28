"""Creative Loop：Insightrackr 式移动水印跟随处理模块（自包含、按需启用）。

移植自 Insightrackr 一键全自动下载与自适应精修工作流中的移动水印处理方式：

- 检测（semantic_brand_analyzer.py / build_verified_brand_override.py）
    - 稠密采样（默认 1.0s 粗扫 + 候选时间窗内 0.2s 精扫），原图 + CLAHE 变体各跑一次 OCR；
    - 只接受与已验证来源产品名称模糊匹配（序列相似度 >= 0.68）的文本；
    - 移动水印轨迹至少 3 个高置信观测点、均值身份置信度 >= 0.62、
      位移 >= 0.018 * min(width, height)、非纯角落，且相邻点时间间隔不超过
      max_interpolation_gap_seconds（默认 0.32s）。
- 跟随渲染（workflow_runner.py _interpolated_waypoint_bbox / _overlay_asset、
  render_manual_brand_fix.py position_at / alpha_overlay）
    - 只在来源水印被连续观测到的时间窗内插值跟随，绝不向首尾外推；
    - 用 RGBA 我方水印素材逐帧叠加，先以柔和底板（0.88~0.92，按置信度自适应）
      压住原标识，再以自适应透明度（0.76~0.82，按置信度自适应）呈现我方水印。
- 逐动作回执（workflow_bridge.py）
    - 每个已检测轨迹必须拿到渲染回执（至少渲染 1 帧）；
    - actions_detected == actions_rendered 是默认硬门禁，缺失即标记失败。

本模块保持自包含：不修改 branding_v09.py 的既有渲染管线，默认关闭、按需启用；
仅复用其受控的 workspace 路径、tesseract OCR、ffmpeg x264 写入器等基础设施。
"""

from __future__ import annotations

import asyncio
import difflib
import hashlib
import json
import math
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from fastapi import HTTPException
from pydantic import BaseModel, Field

try:
    from . import branding_v09 as _operator
except ImportError:  # pragma: no cover - permits direct module debugging
    import branding_v09 as _operator

MODULE_VERSION = "0.1.0"

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class MovingWatermarkFollowRequest(BaseModel):
    # --- source ---
    relative_path: str = Field(
        description="工作区内源视频相对路径（如 raw/2026-08-16_0512/ReelShort/Indonesian/x_x.mp4）"
    )
    competitor_name: str = Field(
        min_length=1, max_length=120,
        description="已验证的来源产品名称（如 ReelShort），用于 OCR 身份匹配"
    )
    aliases: list[str] = Field(
        default_factory=list,
        description="额外别名（如 Reel Short / Reelshort），合并参与匹配",
    )

    # --- legacy own-brand fields (accepted for request compatibility only) ---
    # Dynamic watermark treatment is blur-only. These fields are deliberately
    # ignored by the renderer so a historical client cannot add a brand plate
    # along an observed source-watermark track.
    watermark_asset_relative_path: Optional[str] = Field(
        default=None,
        description="兼容旧请求的保留字段；动态水印仅局部模糊，不会使用该素材。",
    )
    own_brand_text: str = Field(
        default="", max_length=40,
        description="兼容旧请求的保留字段；动态水印仅局部模糊，不会渲染该文本。",
    )

    # --- detection sampling ---
    sample_interval_seconds: float = Field(
        default=1.0, ge=0.25, le=10.0,
        description="粗扫采样间隔（秒）；Insightrackr 对有界长视频约 1s 一级",
    )
    refine_interval_seconds: float = Field(
        default=0.20, ge=0.05, le=1.0,
        description="候选时间窗内精扫间隔（秒），用于把短暂出现的移动水印补成 >= 3 点的真实轨迹",
    )
    refine_window_pad_seconds: float = Field(
        default=2.0, ge=0.0, le=20.0,
        description="候选时间窗两侧外扩用于精扫的秒数",
    )
    max_total_sample_frames: int = Field(
        default=420, ge=20, le=2400,
        description="全片 OCR 采样帧总数上限（粗扫 + 精扫合计）",
    )
    ocr_variants: list[str] = Field(
        default_factory=lambda: ["original", "clahe"],
        description="OCR 预处理变体；半透明浮水印通常在 CLAHE 增强下更易检出",
    )
    ocr_language: str = Field(default="eng", max_length=24)
    ocr_min_confidence: float = Field(
        default=50.0, ge=0.0, le=100.0,
        description="tesseract 置信度下限（0-100 刻度）",
    )
    ocr_scale: float = Field(default=1.5, ge=1.0, le=3.0)

    # --- strict track policy (Insightrackr 参数) ---
    min_observations: int = Field(default=3, ge=3, le=12)
    min_candidate_confidence: float = Field(
        default=0.65, ge=0.0, le=1.0,
        description="进入轨迹构建的候选身份置信度下限（0-1）",
    )
    min_mean_confidence: float = Field(default=0.62, ge=0.0, le=1.0)
    min_movement_ratio: float = Field(default=0.018, ge=0.0, le=0.5)
    max_interpolation_gap_seconds: float = Field(default=0.32, ge=0.10, le=2.0)

    # --- follow render ---
    # Detection-first safety default: this endpoint produces a trajectory
    # report unless a later, explicit review authorizes rendering.
    render_enabled: bool = Field(default=False)
    # Detection-only visual proof. The output contains the untouched source
    # pixels plus colored rectangles for the detected source watermark tracks;
    # it never draws an own-brand asset.
    box_preview_enabled: bool = Field(default=True)
    output_dir_relative_path: Optional[str] = Field(
        default=None,
        description="输出目录（工作区内相对路径）；缺省为 review/moving_watermark_follow/<时间戳>/<源文件名>/",
    )
    cover_expand_ratio: float = Field(default=1.16, ge=1.0, le=2.5)
    blur_sigma: float = Field(
        default=5.0, ge=0.8, le=32.0,
        description="跟随轨迹的局部高斯模糊强度；只影响已验证水印框及其羽化边缘。",
    )
    watermark_width_ratio: float = Field(default=0.42, ge=0.05, le=0.9)
    watermark_height_ratio: float = Field(default=0.105, ge=0.02, le=0.5)
    backdrop_opacity_base: float = Field(default=0.88, ge=0.0, le=1.0)
    backdrop_opacity_span: float = Field(default=0.04, ge=0.0, le=0.12)
    watermark_opacity_base: float = Field(default=0.76, ge=0.0, le=1.0)
    watermark_opacity_span: float = Field(default=0.06, ge=0.0, le=0.12)
    crf: int = Field(default=17, ge=0, le=51)
    preset: str = Field(default="veryfast", max_length=24)

    # --- receipts / artifacts ---
    require_receipt_equality: bool = Field(
        default=True,
        description="硬门禁：检测到的移动水印动作数必须等于实际渲染动作数，否则任务标记失败",
    )
    save_debug_frames: bool = Field(default=True)
    save_contact_sheet: bool = Field(default=True)


class DynamicWatermarkBoxPreviewRequest(BaseModel):
    """Detection-only review request for a reviewed translucent source mark.

    This deliberately has no asset, replacement or render fields.  It is a
    separate endpoint so a reviewer cannot accidentally turn a trajectory
    check into a source-watermark replacement job.
    """

    relative_path: str
    competitor_name: str = Field(default="ReelShort", min_length=1, max_length=120)
    source_brand_id: Optional[str] = None
    source_product_name: Optional[str] = None
    source_app_title: Optional[str] = None
    source_icon_relative_path: Optional[str] = None
    source_logo_relative_path: Optional[str] = None
    sample_interval_seconds: float = Field(default=0.20, ge=0.02, le=1.0)
    # At the default 0.2s visual cadence, 0.45s allows exactly one missed
    # template peak inside an otherwise continuous wordmark pass.  A larger
    # pause remains a hard stop, so a jump/reappearance never becomes a line
    # drawn across the screen.
    max_interpolation_gap_seconds: float = Field(default=0.45, ge=0.10, le=1.0)
    min_template_score: float = Field(default=0.14, ge=0.0, le=1.0)
    min_persistence_ratio: float = Field(default=0.72, ge=0.0, le=1.0)
    output_dir_relative_path: Optional[str] = None


class DynamicWatermarkFadePreviewRequest(BaseModel):
    """Detection-backed preview that only fades the source watermark."""

    relative_path: str
    competitor_name: str = Field(default="ReelShort", min_length=1, max_length=120)
    # Product-specific identity is required for dynamic marks: each source
    # product has a different logo/icon and must not be matched against the
    # generic ReelShort reference.
    source_brand_id: Optional[str] = None
    source_product_name: Optional[str] = None
    source_app_title: Optional[str] = None
    source_icon_relative_path: Optional[str] = None
    source_logo_relative_path: Optional[str] = None
    sample_interval_seconds: float = Field(default=1.0 / 30.0, ge=0.02, le=1.0)
    fade_strength: float = Field(default=0.96, ge=0.0, le=1.0)
    output_dir_relative_path: Optional[str] = None


class DynamicWatermarkTemporalRepairRequest(BaseModel):
    """Temporal clean-repair request; never overwrites the source video."""

    relative_path: str
    competitor_name: str = Field(default="ReelShort", min_length=1, max_length=120)
    source_brand_id: Optional[str] = None
    source_product_name: Optional[str] = None
    source_app_title: Optional[str] = None
    source_icon_relative_path: Optional[str] = None
    source_logo_relative_path: Optional[str] = None
    sample_interval_seconds: float = Field(default=1.0 / 30.0, ge=0.02, le=1.0)
    search_radius_frames: int = Field(default=25, ge=5, le=90)
    max_reference_frames: int = Field(default=5, ge=1, le=12)
    recovery_strength: float = Field(default=1.0, ge=0.0, le=1.0)
    # Never quietly fall back to a blur/fade render and call it inpainting.
    # If valid clean-reference evidence is unavailable, preserve the source and
    # return an explicit review result rather than damaging the story pixels.
    require_clean_reference: bool = True
    require_propainter: bool = True
    max_residual_edge_ratio: float = Field(default=0.32, ge=0.05, le=1.0)
    # Source intervals replaced later by editorial media do not enter the
    # delivered candidate, so their diagnostic tracks must not block QA for the
    # retained story. Intervals are [start_seconds, end_seconds].
    qa_excluded_intervals: list[tuple[float, float]] = Field(default_factory=list)
    # The watermark census is the authority for a production repair.  Passing
    # its already verified, normalized trajectories avoids a second detector
    # pass choosing a different (or empty) set of boxes at render time.
    verified_tracks: list[dict[str, Any]] = Field(default_factory=list)
    verified_tracks_source: Optional[str] = None
    output_dir_relative_path: Optional[str] = None


# ---------------------------------------------------------------------------
# 文本身份匹配（移植 inspect_source_watermark_track.py / semantic_brand_logic.py）
# ---------------------------------------------------------------------------


def _compact(value: Any) -> str:
    return "".join(char for char in str(value).casefold() if char.isalnum())


def _similarity(a: str, b: str) -> float:
    return float(difflib.SequenceMatcher(None, str(a), str(b)).ratio())


def _likely_product(text: Any, aliases: list[str]) -> tuple[bool, float]:
    """与 Insightrackr 一致：包含关系或序列相似度 >= 0.68 才算身份匹配。"""
    candidate = _compact(text)
    if len(candidate) < 3:
        return False, 0.0
    best = 0.0
    for alias in aliases:
        key = _compact(alias)
        if not key:
            continue
        if key in candidate or candidate in key:
            best = max(best, 1.0)
            continue
        ratio = _similarity(candidate, key)
        if ratio >= 0.68:
            best = max(best, ratio)
    return best >= 0.68, round(best, 4)


def _product_aliases(competitor_name: str, extra: list[str]) -> list[str]:
    names = [str(competitor_name or "").strip()]
    for value in extra or []:
        value = str(value or "").strip()
        if value:
            names.append(value)
    result: list[str] = []
    seen = set()
    for name in names:
        if not name:
            continue
        key = _compact(name)
        if key and key not in seen:
            seen.add(key)
            result.append(name)
    return result or ["ReelShort"]


# ---------------------------------------------------------------------------
# 采样与 OCR 观测
# ---------------------------------------------------------------------------


def _normalize_box(x: float, y: float, w: float, h: float, width: int, height: int) -> list[float]:
    return [
        max(0.0, min(1.0, float(x) / max(1, width))),
        max(0.0, min(1.0, float(y) / max(1, height))),
        max(0.0, min(1.0, float(x + w) / max(1, width))),
        max(0.0, min(1.0, float(y + h) / max(1, height))),
    ]


def _variant_bgr(frame: np.ndarray, variant: str) -> np.ndarray:
    if variant == "original":
        return frame
    if variant == "clahe":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        return cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)
    raise HTTPException(status_code=422, detail=f"unsupported_ocr_variant:{variant}")


def _frame_ocr_hits(
    frame: np.ndarray,
    *,
    width: int,
    height: int,
    aliases: list[str],
    language: str,
    min_conf: float,
    scale: float,
    variants: list[str],
) -> list[dict[str, Any]]:
    """对一帧做多个预处理变体的 OCR，返回与来源产品名匹配的观测。"""
    hits: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[float, ...]]] = set()
    for variant in variants or ["original"]:
        try:
            words = _operator._tesseract_words(_variant_bgr(frame, variant), language, min_conf, scale)
        except Exception:
            continue
        for word in words:
            text = str(word.get("text") or "").strip()
            if not text:
                continue
            matched, ratio = _likely_product(text, aliases)
            if not matched:
                continue
            box = _normalize_box(
                float(word.get("x") or 0.0),
                float(word.get("y") or 0.0),
                float(word.get("width") or 1.0),
                float(word.get("height") or 1.0),
                width,
                height,
            )
            if box[2] - box[0] < 0.004 or box[3] - box[1] < 0.004:
                continue
            key = (_compact(text), tuple(round(v, 4) for v in box))
            if key in seen:
                continue
            seen.add(key)
            conf = max(0.0, min(100.0, float(word.get("conf") or 0.0)))
            hits.append(
                {
                    "text": text,
                    "similarity": ratio,
                    "ocr_confidence": round(conf / 100.0, 4),
                    "identity_confidence": round(max(ratio, conf / 100.0), 4),
                    "bbox": box,
                    "variant": variant,
                }
            )
    return hits


def _best_hit_per_time(hits_by_time: dict[float, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """同一秒原图/增强图可重复命中：每秒只保留最高身份置信度的观测。"""
    rows = []
    for second in sorted(hits_by_time):
        group = hits_by_time[second]
        if not group:
            continue
        best = max(group, key=lambda row: float(row["identity_confidence"]))
        cx = (float(best["bbox"][0]) + float(best["bbox"][2])) / 2.0
        cy = (float(best["bbox"][1]) + float(best["bbox"][3])) / 2.0
        rows.append(
            {
                "t": round(float(second), 3),
                "x": round(cx, 5),
                "y": round(cy, 5),
                "bbox": [round(v, 5) for v in best["bbox"]],
                "text": best["text"],
                "confidence": float(best["identity_confidence"]),
                "ocr_confidence": float(best["ocr_confidence"]),
                "similarity": float(best["similarity"]),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# 严格移动水印轨迹（移植 semantic_brand_analyzer.dynamic_watermarks 与
# build_verified_brand_override 的相邻支撑 + 位移 + 置信度 + 非角落门禁）
# ---------------------------------------------------------------------------


def _is_corner_center(x: float, y: float) -> bool:
    return y < 0.17 and (x < 0.38 or x > 0.62)


def _adjacency_supported(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """只保留在相邻 4 秒内出现、且垂直位置相近（|dy| <= 0.10）的同类候选。

    单帧字幕/OCR 误命中不能成为动态水印轨迹；也避免把不同场景里的剧情台词
    串成一条“移动水印”。
    """
    supported = []
    for index, row in enumerate(candidates):
        y = float(row["y"])
        support = [
            other
            for other in candidates
            if other is not row
            and abs(float(other["t"]) - float(row["t"])) <= 4.0
            and abs(float(other["y"]) - y) <= 0.10
        ]
        if support:
            supported.append(row)
    return supported


def _moving_tracks_from_rows(
    rows: list[dict[str, Any]],
    width: int,
    height: int,
    policy: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """把稠密身份观测压缩为严格移动水印轨迹 + 拒绝原因清单。"""
    tracks: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    min_candidate_conf = float(policy["min_candidate_confidence"])
    min_observations = max(3, int(policy["min_observations"]))
    max_gap = float(policy["max_interpolation_gap_seconds"])
    min_motion_px = min(int(width), int(height)) * float(policy["min_movement_ratio"])
    min_mean_conf = float(policy["min_mean_confidence"])

    candidates = [
        row
        for row in rows
        if float(row["confidence"]) >= min_candidate_conf
    ]
    candidates = _adjacency_supported(candidates)
    if len(candidates) < min_observations:
        rejected.append(
            {
                "target": policy["target"],
                "reason": "insufficient_dense_verified_observations",
                "observation_count": len(candidates),
                "minimum": min_observations,
            }
        )
        return tracks, rejected

    ordered = sorted(candidates, key=lambda row: float(row["t"]))
    segments: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for row in ordered:
        if current and float(row["t"]) - float(current[-1]["t"]) > max_gap + 1e-6:
            segments.append(current)
            current = []
        current.append(row)
    if current:
        segments.append(current)

    for segment_index, segment in enumerate(segments, 1):
        if len(segment) < min_observations:
            rejected.append(
                {
                    "target": policy["target"],
                    "reason": "insufficient_dense_verified_observations",
                    "observation_count": len(segment),
                    "minimum": min_observations,
                }
            )
            continue
        xs = [float(row["x"]) * int(width) for row in segment]
        ys = [float(row["y"]) * int(height) for row in segment]
        movement_px = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
        mean_conf = sum(float(row["confidence"]) for row in segment) / len(segment)
        corner_only = all(_is_corner_center(float(row["x"]), float(row["y"])) for row in segment)
        if corner_only:
            rejected.append(
                {
                    "target": policy["target"],
                    "reason": "fixed_corner_identity_not_dynamic",
                    "observation_count": len(segment),
                }
            )
            continue
        if movement_px < min_motion_px:
            rejected.append(
                {
                    "target": policy["target"],
                    "reason": "movement_below_verified_dynamic_threshold",
                    "movement_px": round(movement_px, 2),
                    "minimum_movement_px": round(min_motion_px, 2),
                }
            )
            continue
        if mean_conf < min_mean_conf:
            rejected.append(
                {
                    "target": policy["target"],
                    "reason": "mean_identity_confidence_below_threshold",
                    "mean_identity_confidence": round(mean_conf, 4),
                }
            )
            continue
        tracks.append(
            {
                "track_id": f"insight-moving-{segment_index:02d}",
                "classification": "verified_moving_competitor_watermark",
                "handler": "insightrackr_dense_ocr_follow_track",
                "identity_evidence": "configured_competitor_target_ocr",
                "target": policy["target"],
                "point_count": len(segment),
                "movement_px": round(movement_px, 2),
                "mean_identity_confidence": round(mean_conf, 4),
                "visibility_window": [round(float(segment[0]["t"]), 3), round(float(segment[-1]["t"]), 3)],
                "max_interpolation_gap_seconds": max_gap,
                "tracking_policy": "verified_identity_only__no_pre_or_post_extrapolation__hide_on_tracking_gap",
                "waypoints": segment,
            }
        )
    return tracks, rejected


def _candidate_time_windows(rows: list[dict[str, Any]], pad: float) -> list[list[float]]:
    """从粗扫观测中找出可能包含移动水印的时间窗，供精扫补点。"""
    candidates = sorted(
        [row for row in rows if float(row["confidence"]) >= 0.60 and not _is_corner_center(float(row["x"]), float(row["y"]))],
        key=lambda row: float(row["t"]),
    )
    windows: list[list[float]] = []
    current: list[float] = []
    for row in candidates:
        t = float(row["t"])
        if current and t - current[-1] > 5.0:
            windows.append(current)
            current = []
        current.append(t)
    if current:
        windows.append(current)
    return [
        [max(0.0, float(w[0]) - pad), float(w[-1]) + pad]
        for w in windows
        if len(w) >= 2
    ]


# ---------------------------------------------------------------------------
# 跟随渲染（移植 _interpolated_waypoint_bbox / alpha_overlay / _overlay_asset）
# ---------------------------------------------------------------------------


def _interpolated_waypoint_bbox(waypoints: list[dict[str, Any]], second: float, max_gap: float = 0.32):
    """只在原水印被连续观测到的时间窗内插值，绝不向首尾外推。"""
    rows = sorted(
        [row for row in waypoints if isinstance(row.get("bbox"), list) and len(row["bbox"]) == 4],
        key=lambda row: float(row.get("t", 0.0)),
    )
    if not rows:
        return None
    # Track timestamps are persisted to milliseconds while decoded frame time
    # is an exact rational (for example 28.333333… at 30fps).  A 1ms boundary
    # allowance prevents the final verified frame from disappearing solely due
    # to that representation mismatch; it is far smaller than one video frame.
    boundary_tolerance = 0.001
    if second < float(rows[0]["t"]) - boundary_tolerance or second > float(rows[-1]["t"]) + boundary_tolerance:
        return None
    for left, right in zip(rows, rows[1:]):
        start, end = float(left.get("t", 0.0)), float(right.get("t", 0.0))
        if start <= second <= end:
            if end - start > max_gap:
                return None
            ratio = 0.0 if end <= start else (second - start) / (end - start)
            return [float(a) + (float(b) - float(a)) * ratio for a, b in zip(left["bbox"], right["bbox"])]
    if any(abs(second - float(row.get("t", 0.0))) <= boundary_tolerance for row in rows):
        return list(min(rows, key=lambda row: abs(second - float(row.get("t", 0.0))))["bbox"])
    return None


def _adaptive_opacities(confidence: float, req: MovingWatermarkFollowRequest) -> tuple[float, float]:
    """Insightrackr 文档策略：底板 0.88~0.92、我方水印主体 0.76~0.82，按置信度自适应。"""
    conf = max(0.0, min(1.0, float(confidence)))
    backdrop = max(0.0, min(1.0, float(req.backdrop_opacity_base) + float(req.backdrop_opacity_span) * conf))
    watermark = max(0.0, min(1.0, float(req.watermark_opacity_base) + float(req.watermark_opacity_span) * conf))
    return backdrop, watermark


def _make_own_text_plate(text: str, width_px: int, height_px: int) -> np.ndarray:
    """无 RGBA 素材时，生成透明的我方品牌文本板（白字，供跟随叠加）。"""
    plate = np.zeros((max(8, height_px), max(8, width_px), 4), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, min(3, round(height_px / 18)))
    font_scale = max(0.35, min(2.2, height_px / 26.0))
    (tw, th), baseline = cv2.getTextSize(str(text), font, font_scale, thickness)
    tx = max(0, (plate.shape[1] - tw) // 2)
    ty = max(th, (plate.shape[0] + th) // 2)
    cv2.putText(plate, str(text), (tx, ty), font, font_scale, (255, 255, 255, 255), thickness, cv2.LINE_AA)
    return plate


def _alpha_overlay_follow(
    frame: np.ndarray,
    asset: Optional[np.ndarray],
    center_x: float,
    center_y: float,
    width: int,
    height: int,
    backdrop_opacity: float,
    watermark_opacity: float,
    prepared_asset: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> None:
    """按渲染中心叠加 RGBA 素材：柔和底板压住原标识，再以自适应透明度呈现我方水印。"""
    if width < 4 or height < 4:
        return
    if asset is None:
        asset = _make_own_text_plate("", 1, 1)
    if prepared_asset is None:
        resized = cv2.resize(asset, (width, height), interpolation=cv2.INTER_AREA)
        rgb = resized[:, :, :3].astype(np.float32)
        if resized.shape[2] == 4:
            alpha = resized[:, :, 3:4].astype(np.float32) / 255.0
        else:
            alpha = np.ones((height, width, 1), dtype=np.float32)
        visible = alpha[:, :, 0] > 0.08
        base = np.median(rgb[visible], axis=0) if np.any(visible) else np.array([22.0, 14.0, 86.0], dtype=np.float32)
    else:
        rgb, alpha, base = prepared_asset

    desired_x1 = int(round(center_x - width / 2.0))
    desired_y1 = int(round(center_y - height / 2.0))
    fx1, fy1 = max(0, desired_x1), max(0, desired_y1)
    fx2, fy2 = min(frame.shape[1], desired_x1 + width), min(frame.shape[0], desired_y1 + height)
    if fx2 <= fx1 or fy2 <= fy1:
        return
    ox1, oy1 = fx1 - desired_x1, fy1 - desired_y1
    ox2, oy2 = ox1 + (fx2 - fx1), oy1 + (fy2 - fy1)
    clipped_rgb = rgb[oy1:oy2, ox1:ox2]
    clipped_alpha = alpha[oy1:oy2, ox1:ox2]
    roi = frame[fy1:fy2, fx1:fx2].astype(np.float32)

    backdrop = float(max(0.0, min(1.0, backdrop_opacity)))
    if backdrop > 0:
        roi = np.clip(base, 0, 255)[None, None, :] * backdrop + roi * (1.0 - backdrop)

    op = float(max(0.0, min(1.0, watermark_opacity)))
    frame[fy1:fy2, fx1:fx2] = (clipped_rgb * clipped_alpha * op + roi * (1.0 - clipped_alpha * op)).astype(np.uint8)


def _blur_follow_region(
    frame: np.ndarray,
    center_x: float,
    center_y: float,
    width: int,
    height: int,
    sigma: float,
) -> np.ndarray:
    """Apply the production blur-only policy to one tracked watermark box.

    ``_alpha_overlay_follow`` remains in this module solely for decoding old
    review artifacts.  Delivery rendering must use this helper so that neither
    a supplied asset nor a fallback product name can become a moving/fixed
    brand overlay in the source video.
    """
    if width < 2 or height < 2:
        return frame
    return _operator._operator_apply_feathered_local_blur(
        frame,
        {
            "x": int(round(center_x - width / 2.0)),
            "y": int(round(center_y - height / 2.0)),
            "width": int(width),
            "height": int(height),
        },
        float(sigma),
    )


def _track_overlay_rect(track: dict[str, Any], second: float, width: int, height: int, expand_ratio: float, max_gap: float):
    bbox = _interpolated_waypoint_bbox(list(track.get("waypoints") or []), second, max_gap)
    if not bbox:
        return None, None
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    bw = (bbox[2] - bbox[0]) * float(expand_ratio)
    bh = (bbox[3] - bbox[1]) * float(expand_ratio)
    px = cx * width
    py = cy * height
    return (px, py, max(4, round(bw * width)), max(4, round(bh * height))), bbox


def _bbox_iou_norm(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(1e-9, ua)


def _render_follow_cover(
    video_path: Path,
    out_path: Path,
    tracks: list[dict[str, Any]],
    asset: Optional[np.ndarray],
    req: MovingWatermarkFollowRequest,
    meta: dict[str, Any],
) -> dict[str, Any]:
    """Render verified dynamic tracks with bounded feathered blur only.

    The function name is retained for API compatibility.  It must not draw an
    own-brand asset, text plate, backdrop, or any other replacement layer.
    """
    width = int(meta["width"])
    height = int(meta["height"])
    fps = float(meta["fps"])
    duration = float(meta["duration_seconds"])

    writer, runtime = _operator._open_x264_raw_writer(out_path, width, height, fps, int(req.crf), str(req.preset))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise HTTPException(status_code=422, detail="Could not open source video.")

    started = time.perf_counter()
    per_track_frames: dict[str, dict[str, Any]] = {
        track["track_id"]: {"track_id": track["track_id"], "frames_rendered": 0, "first_render_time": None, "last_render_time": None}
        for track in tracks
    }
    total_cover_frames = 0
    total_applications = 0
    frames_written = 0
    write_error = None
    track_windows = []
    for track in tracks:
        rows = list(track.get("waypoints") or [])
        if rows:
            start_t = min(float(row.get("t", 0.0)) for row in rows)
            end_t = max(float(row.get("t", 0.0)) for row in rows)
        else:
            start_t = end_t = 0.0
        gap = float(track.get("max_interpolation_gap_seconds") or 0.32)
        track_windows.append((track, max(0.0, start_t - gap), min(duration, end_t + gap)))

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            t = frames_written / max(1e-6, fps)
            applied_regions: list[list[float]] = []
            for track, window_start, window_end in track_windows:
                if t < window_start or t > window_end:
                    continue
                rect, bbox = _track_overlay_rect(
                    track, t, width, height, float(req.cover_expand_ratio), float(track.get("max_interpolation_gap_seconds") or 0.32)
                )
                if not rect or not bbox:
                    continue
                if any(_bbox_iou_norm(bbox, prior) >= 0.55 for prior in applied_regions):
                    continue
                applied_regions.append(bbox)
                px, py, pw, ph = rect
                # ``asset`` is intentionally ignored.  The legacy renderer
                # used it (or own_brand_text) to draw a follow-cover plate,
                # which produced the intrusive repeated brand labels reported
                # in delivery frames.  Blur only the small verified region.
                frame = _blur_follow_region(
                    frame,
                    px,
                    py,
                    pw,
                    ph,
                    float(req.blur_sigma),
                )
                total_cover_frames += 1
                total_applications += 1
                receipt = per_track_frames[track["track_id"]]
                receipt["frames_rendered"] += 1
                receipt["first_render_time"] = t if receipt["first_render_time"] is None else receipt["first_render_time"]
                receipt["last_render_time"] = t
            try:
                writer.stdin.write(np.ascontiguousarray(frame).tobytes())
                frames_written += 1
            except (BrokenPipeError, OSError) as exc:
                write_error = f"writer_pipe_error:{exc}"
                break
    finally:
        cap.release()

    try:
        writer.stdin.close()
    except Exception:
        pass
    try:
        return_code = writer.wait(timeout=120)
    except Exception as exc:
        return_code = -1
        write_error = write_error or f"writer_wait_error:{exc}"

    return {
        "output_relative_path": str(out_path.relative_to(_operator.WORKSPACE)).replace("\\", "/"),
        "wall_elapsed_seconds": round(time.perf_counter() - started, 3),
        "frames_written": frames_written,
        "dynamic_cover_frames": total_cover_frames,
        "dynamic_cover_applications": total_applications,
        "treatment": "feathered_gaussian_blur",
        "blur_sigma": float(req.blur_sigma),
        "own_brand_overlay": False,
        "codec": "libx264",
        "crf": int(req.crf),
        "preset": str(req.preset),
        "output_duration_seconds": round(frames_written / max(1e-6, fps), 3),
        "return_code": return_code,
        "write_error": write_error,
        "per_track_receipts": [per_track_frames[track["track_id"]] for track in tracks],
    }


def _render_detection_box_preview(
    video_path: Path,
    out_path: Path,
    tracks: list[dict[str, Any]],
    meta: dict[str, Any],
) -> dict[str, Any]:
    """Write a detection-only MP4 with source-watermark boxes on every frame."""
    width, height = int(meta["width"]), int(meta["height"])
    fps, duration = float(meta["fps"]), float(meta.get("duration_seconds") or 0.0)
    # This is a disposable review artifact, not a delivery encode.  Ultrafast
    # avoids letting the H.264 encoder dominate a detection-only run while the
    # rectangle geometry remains pixel-accurate for manual verification.
    writer, _ = _operator._open_x264_raw_writer(out_path, width, height, fps, 23, "ultrafast")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise HTTPException(status_code=422, detail="Could not open source video for box preview.")
    frames_written = 0
    boxes_drawn = 0
    started = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            second = frames_written / max(1e-6, fps)
            for track_index, track in enumerate(tracks, 1):
                bbox = _interpolated_waypoint_bbox(
                    list(track.get("waypoints") or []),
                    second,
                    float(track.get("max_interpolation_gap_seconds") or 0.32),
                )
                if not bbox:
                    continue
                x1 = max(0, min(width - 1, int(round(float(bbox[0]) * width))))
                y1 = max(0, min(height - 1, int(round(float(bbox[1]) * height))))
                x2 = max(x1 + 1, min(width - 1, int(round(float(bbox[2]) * width))))
                y2 = max(y1 + 1, min(height - 1, int(round(float(bbox[3]) * height))))
                color = (0, 220, 255) if track_index % 2 else (0, 165, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
                label = f"SOURCE WATERMARK {track.get('track_id', track_index)}  {second:.2f}s"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
                ly = max(th + 6, y1)
                cv2.rectangle(frame, (x1, ly - th - 6), (min(width - 1, x1 + tw + 8), ly), color, -1)
                cv2.putText(frame, label, (x1 + 4, ly - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (15, 15, 15), 1, cv2.LINE_AA)
                boxes_drawn += 1
            writer.stdin.write(np.ascontiguousarray(frame).tobytes())
            frames_written += 1
    finally:
        cap.release()
        try:
            writer.stdin.close()
        except Exception:
            pass
    return_code = writer.wait(timeout=180)
    # The raw x264 writer intentionally receives video frames only.  If we
    # hand that intermediate directly to the compositor, FFmpeg sees no source
    # audio and later creates a silent fallback track.  Mux the original audio
    # back onto the repaired video before returning the temporal handoff.
    audio_muxed = False
    audio_mux_error = None
    if return_code == 0 and frames_written > 0:
        mux_path = out_path.with_name(out_path.stem + ".audio-mux.mp4")
        try:
            mux_cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(out_path), "-i", str(source_path),
                "-map", "0:v:0", "-map", "1:a:0?",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
                "-shortest", "-movflags", "+faststart", str(mux_path),
            ]
            subprocess.run(mux_cmd, check=True, timeout=180)
            if mux_path.is_file() and mux_path.stat().st_size > 0:
                out_path.unlink(missing_ok=True)
                mux_path.replace(out_path)
                audio_muxed = True
            else:
                audio_mux_error = "audio_mux_output_missing"
        except Exception as exc:
            audio_mux_error = str(exc)
            try:
                mux_path.unlink(missing_ok=True)
            except Exception:
                pass
    return {
        "enabled": True,
        "output_relative_path": str(out_path.relative_to(_operator.WORKSPACE)).replace("\\", "/"),
        "frames_written": frames_written,
        "boxes_drawn": boxes_drawn,
        "return_code": return_code,
        "wall_elapsed_seconds": round(time.perf_counter() - started, 3),
        "source_pixels_modified": False,
        "own_brand_overlay": False,
    }


def _visual_points_to_preview_tracks(
    tracks: list[dict[str, Any]],
    *,
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    """Convert reviewed-template points to the no-extrapolation preview form."""
    preview_tracks: list[dict[str, Any]] = []
    for track in tracks:
        waypoints: list[dict[str, Any]] = []
        for point in track.get("points") or []:
            bbox = point.get("bbox") or {}
            try:
                x1 = max(0.0, min(1.0, float(bbox["x"]) / width))
                y1 = max(0.0, min(1.0, float(bbox["y"]) / height))
                x2 = max(x1, min(1.0, (float(bbox["x"]) + float(bbox["width"])) / width))
                y2 = max(y1, min(1.0, (float(bbox["y"]) + float(bbox["height"])) / height))
                waypoints.append(
                    {
                        "t": round(float(point["time_seconds"]), 3),
                        "bbox": [x1, y1, x2, y2],
                        "confidence": round(float(point.get("template_score") or 0.0), 4),
                    }
                )
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                continue
        if len(waypoints) >= 2:
            preview_tracks.append(
                {
                    "track_id": str(track.get("track_id") or f"visual-{len(preview_tracks) + 1:02d}"),
                    "visibility_window": list(track.get("visibility_window") or [waypoints[0]["t"], waypoints[-1]["t"]]),
                    "max_interpolation_gap_seconds": float(track.get("max_interpolation_gap_seconds") or 0.32),
                    "waypoints": waypoints,
                    "point_count": len(waypoints),
                    "movement_px": track.get("movement_px"),
                    "identity_evidence": "reviewed_source_watermark_template",
                }
            )
    return preview_tracks


def _localized_alpha_edge_match(
    frame: np.ndarray,
    template_gray: np.ndarray,
    predicted_bbox: dict[str, Any],
    search_radius_px: int = 56,
) -> tuple[float, Optional[dict[str, int]]]:
    """Find the best template edge peak near a trajectory prediction only."""
    if frame is None or frame.size == 0 or template_gray is None or template_gray.size == 0:
        return 0.0, None
    source_height, source_width = frame.shape[:2]
    scale = min(1.0, 360.0 / max(1.0, float(max(source_height, source_width))))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if scale < 0.999:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    template_edges = cv2.Canny(template_gray, 40, 110)
    template_edges = cv2.resize(template_edges, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    scene_edges = cv2.Canny(gray, 40, 110)
    if template_edges.shape[0] >= scene_edges.shape[0] or template_edges.shape[1] >= scene_edges.shape[1]:
        return 0.0, None
    response = cv2.matchTemplate(scene_edges, template_edges, cv2.TM_CCOEFF_NORMED)
    expected_x = int(round(float(predicted_bbox["x"]) * scale))
    expected_y = int(round(float(predicted_bbox["y"]) * scale))
    radius = max(4, int(round(float(search_radius_px) * scale)))
    x1, y1 = max(0, expected_x - radius), max(0, expected_y - radius)
    x2 = min(response.shape[1], expected_x + radius + 1)
    y2 = min(response.shape[0], expected_y + radius + 1)
    if x1 >= x2 or y1 >= y2:
        return 0.0, None
    _, score, _, loc = cv2.minMaxLoc(response[y1:y2, x1:x2])
    return float(score), {
        "x": int(round((x1 + int(loc[0])) / scale)),
        "y": int(round((y1 + int(loc[1])) / scale)),
        "width": max(1, int(round(template_edges.shape[1] / scale))),
        "height": max(1, int(round(template_edges.shape[0] / scale))),
    }


def _recover_localized_visual_bridges(
    video_path: Path,
    tracks: list[dict[str, Any]],
    *,
    width: int,
    height: int,
    sample_interval_seconds: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Join only nearby verified passes with local evidence through a weak gap.

    This is deliberately not unconstrained extrapolation: both gap endpoints
    must have verified template evidence, their velocity must be plausible,
    and every inserted waypoint must obtain a local edge response near the
    predicted path.  A large jump remains two separate tracks.
    """
    if len(tracks) < 2:
        return tracks, []
    paths = list(_operator._dynamic_visual_reference_template_paths())
    template_path = next((path for path in paths if path.name.endswith("_v2.png")), None)
    template_gray = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE) if template_path else None
    if template_gray is None or template_gray.size == 0:
        return tracks, []

    ordered = sorted(tracks, key=lambda track: float((track.get("visibility_window") or [0])[0]))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return tracks, []
    output: list[dict[str, Any]] = []
    bridge_reports: list[dict[str, Any]] = []
    cadence = max(0.10, min(0.50, float(sample_interval_seconds)))
    try:
        current = {**ordered[0], "points": list(ordered[0].get("points") or [])}
        for following_source in ordered[1:]:
            following = {**following_source, "points": list(following_source.get("points") or [])}
            left_points, right_points = current["points"], following["points"]
            if not left_points or not right_points:
                output.append(current)
                current = following
                continue
            left, right = left_points[-1], right_points[0]
            gap = float(right["time_seconds"]) - float(left["time_seconds"])
            left_box, right_box = left["bbox"], right["bbox"]
            left_center = (
                float(left_box["x"]) + float(left_box["width"]) / 2.0,
                float(left_box["y"]) + float(left_box["height"]) / 2.0,
            )
            right_center = (
                float(right_box["x"]) + float(right_box["width"]) / 2.0,
                float(right_box["y"]) + float(right_box["height"]) / 2.0,
            )
            travel = math.dist(left_center, right_center)
            velocity = travel / max(0.001, gap)
            left_area = max(1.0, float(left_box["width"]) * float(left_box["height"]))
            right_area = max(1.0, float(right_box["width"]) * float(right_box["height"]))
            area_ratio = max(left_area, right_area) / min(left_area, right_area)
            eligible = (
                0.45 < gap <= 2.5
                and velocity <= min(width, height) * 0.24
                and area_ratio <= 1.25
            )
            bridge_points: list[dict[str, Any]] = []
            if eligible:
                second = float(left["time_seconds"]) + cadence
                while second < float(right["time_seconds"]) - cadence * 0.35:
                    ratio = (second - float(left["time_seconds"])) / gap
                    predicted = {
                        key: float(left_box[key]) + (float(right_box[key]) - float(left_box[key])) * ratio
                        for key in ("x", "y", "width", "height")
                    }
                    cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000.0)
                    ok, frame = cap.read()
                    score, bbox = _localized_alpha_edge_match(frame, template_gray, predicted) if ok else (0.0, None)
                    if bbox is not None and score >= 0.08:
                        bridge_points.append(
                            {
                                "time_seconds": round(second, 3),
                                "bbox": bbox,
                                "template_score": round(score, 4),
                                "template_relative_path": str(template_path.relative_to(_operator.WORKSPACE)).replace("\\", "/"),
                                "identity_evidence": "localized_low_visibility_template_bridge",
                            }
                        )
                    second += cadence
            expected_count = max(1, int(round((gap - cadence * 0.35) / cadence))) if eligible else 0
            evidence_ratio = len(bridge_points) / max(1, expected_count)
            if eligible and evidence_ratio >= 0.70:
                current["points"].extend(bridge_points)
                current["points"].extend(right_points)
                current["visibility_window"] = [
                    round(float(current["points"][0]["time_seconds"]), 3),
                    round(float(current["points"][-1]["time_seconds"]), 3),
                ]
                current["point_count"] = len(current["points"])
                current["tracking_policy"] = "reviewed_template_identity_with_localized_low_visibility_bridge"
                bridge_reports.append(
                    {
                        "from_track_id": str(current.get("track_id")),
                        "to_track_id": str(following.get("track_id")),
                        "gap_start_seconds": round(float(left["time_seconds"]), 3),
                        "gap_end_seconds": round(float(right["time_seconds"]), 3),
                        "gap_seconds": round(gap, 3),
                        "interpolated_waypoints": len(bridge_points),
                        "localized_evidence_ratio": round(evidence_ratio, 3),
                        "endpoint_velocity_px_per_second": round(velocity, 2),
                    }
                )
                continue
            output.append(current)
            current = following
        output.append(current)
    finally:
        cap.release()
    return output, bridge_reports


def _recover_localized_visual_tails(
    video_path: Path,
    tracks: list[dict[str, Any]],
    *,
    sample_interval_seconds: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Extend a verified track through its final faint frames, frame by frame."""
    paths = list(_operator._dynamic_visual_reference_template_paths())
    template_path = next((path for path in paths if path.name.endswith("_v2.png")), None)
    template_gray = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE) if template_path else None
    if template_gray is None or template_gray.size == 0:
        return tracks, []
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not cap.isOpened() or fps <= 0.0:
        cap.release()
        return tracks, []
    # No more than half a second can be appended without a fresh coarse
    # detection.  The first local miss ends the tail immediately.
    maximum_frames = max(2, min(18, int(round(fps * 0.50))))
    output: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []

    def is_hard_scene_cut(previous_frame: np.ndarray, current_frame: np.ndarray) -> bool:
        if previous_frame is None or current_frame is None:
            return False
        previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        previous_small = cv2.resize(previous_gray, (72, 128), interpolation=cv2.INTER_AREA)
        current_small = cv2.resize(current_gray, (72, 128), interpolation=cv2.INTER_AREA)
        # A hard cut must terminate a fading-mark tail.  This comparison is
        # intentionally only a safety stop; ordinary character motion and
        # subtitle changes remain well below this full-frame difference.
        return float(np.mean(cv2.absdiff(previous_small, current_small))) >= 34.0

    try:
        for source_track in tracks:
            track = {**source_track, "points": list(source_track.get("points") or [])}
            points = track["points"]
            if len(points) < 2:
                output.append(track)
                continue
            previous, last = points[-2], points[-1]
            delta = float(last["time_seconds"]) - float(previous["time_seconds"])
            if delta <= 0.0 or delta > max(0.60, float(sample_interval_seconds) * 2.2):
                output.append(track)
                continue
            # Decode sequentially from the exact source frame. Re-seeking by
            # millisecond can return the preceding frame around a cut, which
            # both delayed the cut guard and made the final fade appear to
            # extend into the next shot.
            base_frame_index = max(0, int(round(float(last["time_seconds"]) * fps)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, base_frame_index)
            _, previous_frame = cap.read()
            recovered: list[dict[str, Any]] = []
            for index in range(1, maximum_frames + 1):
                second = (base_frame_index + index) / fps
                ratio = (second - float(last["time_seconds"])) / delta
                predicted = {
                    key: float(last["bbox"][key]) + (float(last["bbox"][key]) - float(previous["bbox"][key])) * ratio
                    for key in ("x", "y", "width", "height")
                }
                ok, frame = cap.read()
                if ok and is_hard_scene_cut(previous_frame, frame):
                    break
                if ok:
                    previous_frame = frame
                score, bbox = _localized_alpha_edge_match(frame, template_gray, predicted, search_radius_px=48) if ok else (0.0, None)
                if bbox is None or score < 0.12:
                    break
                position_error = math.dist(
                    (float(bbox["x"]), float(bbox["y"])),
                    (float(predicted["x"]), float(predicted["y"])),
                )
                if position_error > 42.0:
                    break
                recovered.append(
                    {
                        "time_seconds": round(second, 3),
                        "bbox": bbox,
                        "template_score": round(score, 4),
                        "template_relative_path": str(template_path.relative_to(_operator.WORKSPACE)).replace("\\", "/"),
                        "identity_evidence": "localized_final_fade_template_track",
                    }
                )
            # A single lucky edge peak is not enough to extend a track.
            if len(recovered) >= 2:
                track["points"].extend(recovered)
                track["visibility_window"] = [
                    round(float(track["points"][0]["time_seconds"]), 3),
                    round(float(track["points"][-1]["time_seconds"]), 3),
                ]
                track["point_count"] = len(track["points"])
                reports.append(
                    {
                        "track_id": str(track.get("track_id")),
                        "from_seconds": round(float(last["time_seconds"]), 3),
                        "to_seconds": round(float(recovered[-1]["time_seconds"]), 3),
                        "frames_recovered": len(recovered),
                    }
                )
            output.append(track)
    finally:
        cap.release()
    return output, reports


def _recover_localized_visual_heads(
    video_path: Path,
    tracks: list[dict[str, Any]],
    *,
    sample_interval_seconds: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Recover faint initial frames before every verified visual trajectory."""
    paths = list(_operator._dynamic_visual_reference_template_paths())
    template_path = next((path for path in paths if path.name.endswith("_v2.png")), None)
    template_gray = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE) if template_path else None
    if template_gray is None or template_gray.size == 0:
        return tracks, []
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not cap.isOpened() or fps <= 0.0:
        cap.release()
        return tracks, []
    maximum_frames = max(2, min(18, int(round(fps * 0.50))))
    output: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    try:
        for source_track in tracks:
            track = {**source_track, "points": list(source_track.get("points") or [])}
            points = track["points"]
            if len(points) < 2:
                output.append(track)
                continue
            first, following = points[0], points[1]
            delta = float(following["time_seconds"]) - float(first["time_seconds"])
            if delta <= 0.0 or delta > max(0.60, float(sample_interval_seconds) * 2.2):
                output.append(track)
                continue
            recovered: list[dict[str, Any]] = []
            for index in range(1, maximum_frames + 1):
                second = float(first["time_seconds"]) - index / fps
                if second < 0.0:
                    break
                ratio = (float(first["time_seconds"]) - second) / delta
                predicted = {
                    key: float(first["bbox"][key]) - (float(following["bbox"][key]) - float(first["bbox"][key])) * ratio
                    for key in ("x", "y", "width", "height")
                }
                cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000.0)
                ok, frame = cap.read()
                score, bbox = _localized_alpha_edge_match(frame, template_gray, predicted, search_radius_px=48) if ok else (0.0, None)
                if bbox is None or score < 0.12:
                    break
                position_error = math.dist(
                    (float(bbox["x"]), float(bbox["y"])),
                    (float(predicted["x"]), float(predicted["y"])),
                )
                if position_error > 42.0:
                    break
                recovered.append(
                    {
                        "time_seconds": round(second, 3),
                        "bbox": bbox,
                        "template_score": round(score, 4),
                        "template_relative_path": str(template_path.relative_to(_operator.WORKSPACE)).replace("\\", "/"),
                        "identity_evidence": "localized_initial_fade_template_track",
                    }
                )
            if len(recovered) >= 2:
                recovered.reverse()
                track["points"] = [*recovered, *track["points"]]
                track["visibility_window"] = [
                    round(float(track["points"][0]["time_seconds"]), 3),
                    round(float(track["points"][-1]["time_seconds"]), 3),
                ]
                track["point_count"] = len(track["points"])
                reports.append(
                    {
                        "track_id": str(track.get("track_id")),
                        "from_seconds": round(float(first["time_seconds"]), 3),
                        "to_seconds": round(float(recovered[0]["time_seconds"]), 3),
                        "frames_recovered": len(recovered),
                    }
                )
            output.append(track)
    finally:
        cap.release()
    return output, reports


def dynamic_watermark_box_preview_sync(req: DynamicWatermarkBoxPreviewRequest) -> dict[str, Any]:
    """Create a source-watermark trajectory review video without OCR or overlay.

    The regular follow endpoint begins with Tesseract and is therefore a poor
    evidence source for a faint alpha-blended wordmark.  This path consumes
    only the reviewed visual reference and never invokes either own-brand
    rendering or the OCR-based replacement handoff.
    """
    started = time.monotonic()
    source_path = _operator._safe_workspace_path(req.relative_path, must_exist=True)
    if source_path.suffix.lower() not in {".mp4", ".mov", ".mkv", ".webm"}:
        raise HTTPException(status_code=422, detail="source must be a video file")

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise HTTPException(status_code=422, detail="Could not open source video.")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    if fps <= 0.0 or width <= 0 or height <= 0:
        raise HTTPException(status_code=422, detail="video metadata unreadable")
    duration = round(frame_count / fps, 3) if frame_count else 0.0

    census_request = _operator.BrandingWatermarkCensusRequest(
        relative_path=str(source_path.relative_to(_operator.WORKSPACE)).replace("\\", "/"),
        targets=[str(req.competitor_name)],
        source_brand_id=req.source_brand_id,
        source_product_name=req.source_product_name,
        source_app_title=req.source_app_title,
        source_icon_relative_path=req.source_icon_relative_path,
        source_logo_relative_path=req.source_logo_relative_path,
        dynamic_visual_mode=True,
        dynamic_visual_sample_interval_seconds=float(req.sample_interval_seconds),
        dynamic_visual_min_persistence_ratio=float(req.min_persistence_ratio),
        # The native v2 crop is an edge reference.  Its score is not an OCR
        # probability, and needs the conservative native score threshold below.
        dynamic_visual_min_alpha_response=float(req.min_template_score),
        dynamic_visual_native_edge_min_score=float(req.min_template_score),
        save_debug_frames=False,
    )
    visual_scan = _operator._dynamic_visual_watermark_scan(
        source_path,
        width=width,
        height=height,
        duration=duration,
        ocr_hits=[],
        req=census_request,
    )
    # Source-specific reviewed icon templates are identity-bound but their
    # semi-transparent matches score lower than the opaque reference crop.
    # Keep the generic reference threshold strict while allowing the calibrated
    # source template range used by the operator handoff.
    source_specific_template = bool(
        (visual_scan or {}).get("source_specific_template_available")
    )
    strict_policy = {
        "min_observations": 3,
        "min_movement_ratio": 0.018,
        "max_interpolation_gap_seconds": float(req.max_interpolation_gap_seconds),
        # App icons supplied by acquisition are often rendered at a much
        # smaller alpha-blended size than the 96px registry asset.  Their
        # structural score is consequently lower than the reviewed generic
        # crop; retain the motion/persistence/anchor gates but do not discard
        # the entire track on the old 0.30 mean-score cutoff.
        "min_visual_mean_score": 0.22 if source_specific_template else 0.45,
        "min_native_edge_mean_score": float(req.min_template_score),
        "min_native_edge_anchor_score": 0.20,
        "min_native_edge_anchor_count": 2,
        "min_native_edge_path_straightness": 0.45,
        "min_visual_persistence_ratio": float(req.min_persistence_ratio),
        "require_three_visual_observations": True,
    }
    strict_tracks, rejected = _operator._strict_verified_visual_tracks(
        visual_scan,
        width,
        height,
        strict_policy,
        duration=duration,
    )
    strict_tracks, localized_bridges = _recover_localized_visual_bridges(
        source_path,
        strict_tracks,
        width=width,
        height=height,
        sample_interval_seconds=float(req.sample_interval_seconds),
    )
    strict_tracks, localized_head_extensions = _recover_localized_visual_heads(
        source_path,
        strict_tracks,
        sample_interval_seconds=float(req.sample_interval_seconds),
    )
    strict_tracks, localized_tail_extensions = _recover_localized_visual_tails(
        source_path,
        strict_tracks,
        sample_interval_seconds=float(req.sample_interval_seconds),
    )
    preview_tracks = _visual_points_to_preview_tracks(strict_tracks, width=width, height=height)

    if req.output_dir_relative_path:
        out_dir = _operator._safe_workspace_path(req.output_dir_relative_path, must_exist=False)
    else:
        out_dir = (
            _operator.WORKSPACE
            / "review"
            / "dynamic_watermark_box_preview"
            / datetime.now(_operator._app_now().tzinfo).strftime("%Y%m%d-%H%M%S")
            / source_path.stem
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "ok": True,
        "status": "completed" if preview_tracks else "no_verified_dynamic_watermark",
        "mode": "reviewed_visual_source_watermark_box_preview",
        "module_version": MODULE_VERSION,
        "source": {
            "relative_path": str(source_path.relative_to(_operator.WORKSPACE)).replace("\\", "/"),
            "duration_seconds": duration,
            "fps": round(fps, 3),
            "width": width,
            "height": height,
            "source_file_modified": False,
        },
        "detector": {
            "identity": "reviewed_visual_template_only",
            "ocr_used": False,
            "sample_interval_seconds": float(req.sample_interval_seconds),
            "strict_policy": strict_policy,
            "template_relative_paths": list(visual_scan.get("scan_template_relative_paths") or []),
            "raw_candidate_track_count": len(visual_scan.get("tracks") or []),
            "verified_track_count": len(preview_tracks),
            "tracks": preview_tracks,
            "localized_low_visibility_bridges": localized_bridges,
            "localized_initial_fade_extensions": localized_head_extensions,
            "localized_final_fade_extensions": localized_tail_extensions,
            "rejected_candidates": rejected,
        },
        "box_preview": {
            "enabled": bool(preview_tracks),
            "output_relative_path": None,
            "frames_written": 0,
            "boxes_drawn": 0,
            "source_pixels_modified": False,
            "own_brand_overlay": False,
        },
        "notes": [
            "Detection-only evidence: only SOURCE WATERMARK rectangles are added to a new review video.",
            "No own-brand image, text, watermark replacement, masking or source-video modification is permitted by this endpoint.",
            "A box is never extrapolated before the first observation or after the last one. Ordinary tracking gaps remain blank.",
            "A short gap between two slow, size-consistent verified passes may be recovered only when localized template evidence follows the predicted path at at least 70% of bridge samples.",
            "A final faint tail is recovered frame by frame for at most half a second, stopping at the first local template miss.",
            "The same frame-level local check is run backward from every verified start to recover a faint initial fade without searching unrelated screen regions.",
        ],
    }
    if preview_tracks:
        preview_path = out_dir / f"{source_path.stem}.source-watermark-box-preview.mp4"
        report["box_preview"].update(
            _render_detection_box_preview(
                source_path,
                preview_path,
                preview_tracks,
                report["source"],
            )
        )

    report["wall_elapsed_seconds"] = round(time.monotonic() - started, 3)
    report_path = out_dir / f"{source_path.stem}.source-watermark-box-preview.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_relative_path"] = str(report_path.relative_to(_operator.WORKSPACE)).replace("\\", "/")
    return report


def _fade_source_watermark_frame(
    frame: np.ndarray,
    tracks: list[dict[str, Any]],
    second: float,
    width: int,
    height: int,
    fade_strength: float,
    temporal_references: Optional[list[tuple[float, np.ndarray]]] = None,
    template_relative_path: Optional[str] = None,
) -> tuple[np.ndarray, int, int]:
    """Repair detected watermark strokes with temporal clean plates.

    Only the reviewed watermark glyph support is changed.  Nearby frames whose
    watermark has moved away provide the primary clean plate, and a small
    Telea inpaint is used only within that glyph mask to close residual outline
    pixels.  This is deliberately not a rectangle blur or a brand overlay.
    """
    output = frame.copy()
    masks_applied = 0
    mask_pixels = 0
    template_cache: Optional[np.ndarray] = None
    template_path = None
    if template_relative_path:
        try:
            candidate = _operator._safe_workspace_path(template_relative_path, must_exist=True)
            if candidate.is_file():
                template_path = candidate
        except Exception:
            template_path = None
    if template_path is None:
        template_path = _operator.WORKSPACE / "config" / "dynamic_watermark_reference_reelshort_v2.png"
    if template_path.exists():
        template_cache = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
    glyph_cache = getattr(_fade_source_watermark_frame, "_glyph_cache", {})
    template_cache_key = None
    if template_cache is not None:
        try:
            template_cache_key = (int(template_path.stat().st_mtime_ns), int(template_cache.shape[0]), int(template_cache.shape[1]))
        except Exception:
            template_cache_key = (int(template_cache.shape[0]), int(template_cache.shape[1]))
    for track in tracks:
        # Most verified tracks are short bursts. Avoid walking every track for
        # every frame; this was the dominant CPU cost on 90–120s clips with
        # dozens of template tracks.
        visibility = track.get("visibility_window") or []
        if len(visibility) == 2:
            try:
                gap = float(track.get("max_interpolation_gap_seconds") or 0.45)
                if float(second) < float(visibility[0]) - gap or float(second) > float(visibility[1]) + gap:
                    continue
            except Exception:
                pass
        normalized = _interpolated_waypoint_bbox(
            list(track.get("waypoints") or []),
            second,
            float(track.get("max_interpolation_gap_seconds") or 0.45),
        )
        if not normalized:
            continue
        bbox = {
            "x": int(round(float(normalized[0]) * width)),
            "y": int(round(float(normalized[1]) * height)),
            "width": max(1, int(round((float(normalized[2]) - float(normalized[0])) * width))),
            "height": max(1, int(round((float(normalized[3]) - float(normalized[1])) * height))),
        }
        stroke = None
        if template_cache is not None:
            glyph_key = (template_cache_key, int(bbox["width"]), int(bbox["height"]))
            glyph = glyph_cache.get(glyph_key)
            if glyph is None:
                gray_t = template_cache
                smooth_t = cv2.GaussianBlur(gray_t, (0, 0), 7.0)
                deviation = cv2.absdiff(gray_t, smooth_t)
                glyph = np.where(deviation >= 15, 255, 0).astype(np.uint8)
                glyph = cv2.morphologyEx(glyph, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
                glyph = cv2.dilate(glyph, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
                glyph = cv2.resize(glyph, (bbox["width"], bbox["height"]), interpolation=cv2.INTER_LINEAR)
                if len(glyph_cache) > 32:
                    glyph_cache.clear()
                glyph_cache[glyph_key] = glyph
            resized = glyph
            full = np.zeros((height, width), dtype=np.uint8)
            x0, y0 = max(0, bbox["x"]), max(0, bbox["y"])
            x1, y1 = min(width, bbox["x"] + bbox["width"]), min(height, bbox["y"] + bbox["height"])
            if x1 > x0 and y1 > y0:
                full[y0:y1, x0:x1] = resized[: y1 - y0, : x1 - x0]
                coverage = float(np.count_nonzero(full)) / max(1.0, float((x1 - x0) * (y1 - y0)))
                if 0.03 <= coverage <= 0.65:
                    stroke = {"mask": full, "coverage_ratio": coverage}
        if stroke is None:
            stroke = _operator._operator_dynamic_wordmark_stroke_mask(output, bbox)
        if stroke is None:
            continue
        raw_mask = stroke["mask"]
        # Close anti-aliased gaps but keep the support well inside the tracked
        # wordmark. No rectangular ROI is ever painted.
        repair_mask = cv2.dilate(
            raw_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=2,
        )
        soft = cv2.GaussianBlur(repair_mask, (0, 0), 1.1).astype(np.float32) / 255.0
        alpha = np.clip(soft * float(fade_strength), 0.0, 1.0)[..., None]
        x0, y0 = max(0, bbox["x"] - 3), max(0, bbox["y"] - 3)
        x1 = min(width, bbox["x"] + bbox["width"] + 3)
        y1 = min(height, bbox["y"] + bbox["height"] + 3)
        if x1 <= x0 or y1 <= y0:
            continue
        local = output[y0:y1, x0:x1]
        candidates = []
        current_cx = (bbox["x"] + bbox["width"] * 0.5) / max(1.0, float(width))
        current_cy = (bbox["y"] + bbox["height"] * 0.5) / max(1.0, float(height))
        current_diag = max(1.0, float((bbox["width"] ** 2 + bbox["height"] ** 2) ** 0.5))
        for ref_second, ref in temporal_references or []:
            if ref is None or ref.shape != output.shape or abs(float(ref_second) - second) < 0.18:
                continue
            ref_norm = _interpolated_waypoint_bbox(
                list(track.get("waypoints") or []),
                float(ref_second),
                float(track.get("max_interpolation_gap_seconds") or 0.45),
            )
            if not ref_norm:
                continue
            ref_cx = (float(ref_norm[0]) + float(ref_norm[2])) * 0.5
            ref_cy = (float(ref_norm[1]) + float(ref_norm[3])) * 0.5
            distance_px = (((ref_cx - current_cx) * width) ** 2 + ((ref_cy - current_cy) * height) ** 2) ** 0.5
            if distance_px >= current_diag * 0.85:
                candidates.append((distance_px, ref))
        repaired_roi = None
        warped_refs = []
        # Align each clean candidate to the current ROI with dense optical
        # flow before combining.  This prevents actor/camera motion from
        # becoming a ghost inside the repaired watermark.
        # A full-resolution Farneback pass for five references on every active
        # frame dominated runtime (hundreds of passes on a 100s clip). Two
        # displaced references are sufficient for a median clean plate; use a
        # half-resolution flow field and upscale it for the final remap.
        for _, ref in sorted(candidates, key=lambda item: item[0], reverse=True)[:2]:
            ref_roi = ref[y0:y1, x0:x1]
            if ref_roi.shape != local.shape:
                continue
            try:
                # Use the established current-to-reference warp. This watermark
                # moves across large face/background changes; inverse sampling
                # generated block-shaped ghosts in visual QC even though it is
                # mathematically appealing for small motion.
                flow_scale = 0.5 if min(local.shape[:2]) >= 96 else 1.0
                if flow_scale < 1.0:
                    flow_src = cv2.resize(
                        cv2.cvtColor(local, cv2.COLOR_BGR2GRAY), None,
                        fx=flow_scale, fy=flow_scale,
                        interpolation=cv2.INTER_AREA,
                    )
                    flow_ref = cv2.resize(
                        cv2.cvtColor(ref_roi, cv2.COLOR_BGR2GRAY),
                        (flow_src.shape[1], flow_src.shape[0]),
                        interpolation=cv2.INTER_AREA,
                    )
                else:
                    flow_src = cv2.cvtColor(local, cv2.COLOR_BGR2GRAY)
                    flow_ref = cv2.cvtColor(ref_roi, cv2.COLOR_BGR2GRAY)
                flow_small = cv2.calcOpticalFlowFarneback(
                    flow_src, flow_ref, None, 0.45, 2, 11, 2, 5, 1.1, 0,
                )
                if flow_scale < 1.0:
                    flow = cv2.resize(
                        flow_small,
                        (local.shape[1], local.shape[0]),
                        interpolation=cv2.INTER_LINEAR,
                    ) / flow_scale
                else:
                    flow = flow_small
                gx, gy = np.meshgrid(np.arange(ref_roi.shape[1], dtype=np.float32), np.arange(ref_roi.shape[0], dtype=np.float32))
                warped = cv2.remap(ref_roi, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                warped_refs.append(warped)
            except Exception:
                continue
        if warped_refs:
            repaired_roi = np.median(np.stack(warped_refs, axis=0), axis=0).astype(np.uint8)
        if repaired_roi is None:
            # The detector can identify the watermark but has no displaced
            # clean reference for this frame.  Repair the compact glyph mask
            # only; never blur the broad tracked ROI as a fallback.
            repaired_roi = local.copy()
        local_mask_u8 = repair_mask[y0:y1, x0:x1]
        local_mask_u8 = np.where(local_mask_u8 >= 96, 255, 0).astype(np.uint8)
        # Use the temporal clean plate outside the glyphs, then explicitly
        # inpaint every glyph pixel.  Alpha blending the original glyph back in
        # leaves an identifiable semi-transparent watermark; do not do that.
        repaired = repaired_roi.copy()
        if np.any(local_mask_u8):
            try:
                # The compact mask bounds the fill operation to watermark
                # strokes, avoiding any broad face/background reconstruction.
                # This is retained after visual QC: omitting it left blocky
                # clean-plate artifacts along the moving glyphs.
                repaired = cv2.inpaint(
                    repaired,
                    local_mask_u8,
                    3.0,
                    cv2.INPAINT_TELEA,
                )
            except Exception:
                pass
        local = output[y0:y1, x0:x1]
        core = local_mask_u8 > 0
        local[core] = repaired[core]
        # Feather only the one-pixel anti-aliased edge, never the full ROI.
        edge_alpha = cv2.GaussianBlur(local_mask_u8, (0, 0), 0.7).astype(np.float32) / 255.0
        edge = (edge_alpha > 0.0) & ~core
        if np.any(edge):
            blend = edge_alpha[..., None]
            mixed = np.clip(
                local.astype(np.float32) * (1.0 - blend)
                + repaired.astype(np.float32) * blend,
                0,
                255,
            ).astype(np.uint8)
            local[edge] = mixed[edge]
        output[y0:y1, x0:x1] = local
        masks_applied += 1
        mask_pixels += int(np.count_nonzero(repair_mask))
    _fade_source_watermark_frame._glyph_cache = glyph_cache
    return output, masks_applied, mask_pixels


def dynamic_watermark_fade_preview_sync(req: DynamicWatermarkFadePreviewRequest) -> dict[str, Any]:
    """Generate a fade-only preview using the verified detection trajectory."""
    started = time.monotonic()
    source_path = _operator._safe_workspace_path(req.relative_path, must_exist=True)
    if source_path.suffix.lower() not in {".mp4", ".mov", ".mkv", ".webm"}:
        raise HTTPException(status_code=422, detail="source must be a video file")

    # Reuse the full boundary-audited detector. Its temporary box artifact is
    # intentionally retained beside the fade preview for before/after review.
    detection_dir = req.output_dir_relative_path or (
        "review/dynamic_watermark_fade_preview/"
        + datetime.now(_operator._app_now().tzinfo).strftime("%Y%m%d-%H%M%S")
        + "/"
        + source_path.stem
    )
    # Detection uses a visual cadence; rendering still processes every frame.
    # Sampling at 30 fps fragments short moving marks and can yield zero strict
    # tracks, even though the same video is detectable at 0.15-0.50 seconds.
    detection_sample_interval = max(0.15, min(0.50, float(req.sample_interval_seconds)))
    detection = dynamic_watermark_box_preview_sync(
        DynamicWatermarkBoxPreviewRequest(
            relative_path=str(source_path.relative_to(_operator.WORKSPACE)).replace("\\", "/"),
            competitor_name=req.competitor_name,
            source_brand_id=req.source_brand_id,
            source_product_name=req.source_product_name,
            source_app_title=req.source_app_title,
            source_icon_relative_path=req.source_icon_relative_path,
            source_logo_relative_path=req.source_logo_relative_path,
            sample_interval_seconds=detection_sample_interval,
            output_dir_relative_path=detection_dir,
        )
    )
    tracks = list(detection.get("detector", {}).get("tracks") or [])
    source_meta = detection["source"]
    out_dir = _operator._safe_workspace_path(detection_dir, must_exist=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{source_path.stem}.source-watermark-fade-preview.mp4"
    writer, _ = _operator._open_x264_raw_writer(out_path, int(source_meta["width"]), int(source_meta["height"]), float(source_meta["fps"]), 23, "ultrafast")
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise HTTPException(status_code=422, detail="Could not open source video.")
    frames_written = 0
    masks_applied = 0
    mask_pixels = 0
    past_frames: list[tuple[float, np.ndarray]] = []
    lookahead: list[tuple[float, np.ndarray]] = []
    try:
        for _ in range(25):
            ok, buffered = cap.read()
            if not ok or buffered is None:
                break
            lookahead.append((len(lookahead) / max(1e-6, float(source_meta["fps"])), buffered))
        while True:
            if not lookahead:
                break
            buffered_second, frame = lookahead.pop(0)
            refs: list[tuple[float, np.ndarray]] = []
            if past_frames:
                refs.extend([past_frames[0], past_frames[len(past_frames) // 2]])
            if lookahead:
                refs.append(lookahead[-1])
            second = frames_written / max(1e-6, float(source_meta["fps"]))
            processed, applied, pixels = _fade_source_watermark_frame(
                frame,
                tracks,
                second,
                int(source_meta["width"]),
                int(source_meta["height"]),
                float(req.fade_strength),
                refs,
                req.source_icon_relative_path or req.source_logo_relative_path,
            )
            writer.stdin.write(np.ascontiguousarray(processed).tobytes())
            past_frames.append((second, frame))
            if len(past_frames) > 25:
                del past_frames[0]
            ok, next_frame = cap.read()
            if ok and next_frame is not None:
                lookahead.append((second + 25.0 / max(1e-6, float(source_meta["fps"])), next_frame))
            frames_written += 1
            masks_applied += applied
            mask_pixels += pixels
    finally:
        cap.release()
        try:
            writer.stdin.close()
        except Exception:
            pass
    return_code = writer.wait(timeout=180)
    report = {
        "ok": return_code == 0 and frames_written > 0,
        "status": "completed" if return_code == 0 and frames_written > 0 else "encode_failed",
        "mode": "dynamic_source_watermark_fade_only_preview",
        "source": source_meta,
        "detection": {
            "verified_track_count": len(tracks),
            "tracks": tracks,
            "box_preview_relative_path": detection.get("box_preview", {}).get("output_relative_path"),
        },
        "fade": {
            "strength": float(req.fade_strength),
            "frames_written": frames_written,
            "frames_with_masks": masks_applied,
            "mask_pixels_applied": mask_pixels,
            "return_code": return_code,
            "audio_muxed_from_source": audio_muxed,
            "audio_mux_error": audio_mux_error,
            "output_relative_path": str(out_path.relative_to(_operator.WORKSPACE)).replace("\\", "/"),
            "source_video_modified": False,
            "own_brand_overlay": False,
            "method": "template_glyph_mask_temporal_clean_plate_flow_telea",
        },
        "notes": [
            "Only detected source-watermark strokes are partially repaired; no own-brand text, logo or replacement plate is drawn.",
            "The source video is never overwritten. Compare this output with the adjacent detection-box preview before approving a stronger fade strength.",
        ],
        "wall_elapsed_seconds": round(time.monotonic() - started, 3),
    }
    report_path = out_dir / f"{source_path.stem}.source-watermark-fade-preview.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_relative_path"] = str(report_path.relative_to(_operator.WORKSPACE)).replace("\\", "/")
    return report


def _template_aligned_high_pass_correlation(
    gray: np.ndarray,
    template_signal: np.ndarray,
    template_support: np.ndarray,
) -> Optional[float]:
    """Measure residual structure that still matches the reviewed template.

    Generic edge energy inside a moving watermark box also contains legitimate
    face, subtitle and background texture. This signed normalized correlation
    gates only the reviewed bright/dark watermark stroke pattern.
    """
    if gray is None or gray.size == 0 or template_signal is None or template_signal.size == 0:
        return None
    if template_support is None or template_support.size == 0:
        return None
    height, width = gray.shape[:2]
    if height < 2 or width < 2:
        return None
    signal = cv2.resize(
        template_signal.astype(np.float32),
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )
    support = cv2.resize(
        template_support.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    if int(np.count_nonzero(support)) < 32:
        return None
    observed = gray.astype(np.float32) - cv2.GaussianBlur(
        gray.astype(np.float32),
        (0, 0),
        2.0,
    )
    expected_values = signal[support]
    observed_values = observed[support]
    expected_values = expected_values - float(np.mean(expected_values))
    observed_values = observed_values - float(np.mean(observed_values))
    denominator = float(
        np.linalg.norm(expected_values) * np.linalg.norm(observed_values)
    )
    if denominator <= 1e-6:
        return 0.0
    return float(np.dot(expected_values, observed_values) / denominator)


def _is_in_editorial_exclusion(second: float, intervals: list[tuple[float, float]]) -> bool:
    """Return whether a source timestamp is replaced before final delivery."""
    return any(float(start) <= second <= float(end) for start, end in intervals)


def _authoritative_temporal_tracks(raw_tracks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate census trajectories without re-detecting the source video.

    The frame-mask worker accepts normalized ``waypoints``.  Keep this small
    validation here so an old/malformed report cannot accidentally mask the
    whole frame, while a valid strict census is used exactly as approved.
    """
    normalized: list[dict[str, Any]] = []
    for index, track in enumerate(raw_tracks or [], 1):
        waypoints = []
        for point in track.get("waypoints") or []:
            bbox = point.get("bbox") or []
            try:
                values = [float(value) for value in bbox]
                second = float(point.get("t"))
            except (TypeError, ValueError):
                continue
            if len(values) != 4:
                continue
            x0, y0, x1, y1 = values
            if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
                continue
            waypoints.append({
                "t": round(second, 3), "bbox": [x0, y0, x1, y1],
                "confidence": float(point.get("confidence") or 0.0),
            })
        waypoints.sort(key=lambda point: point["t"])
        if len(waypoints) < 2:
            continue
        window = track.get("visibility_window") or [waypoints[0]["t"], waypoints[-1]["t"]]
        try:
            start, end = float(window[0]), float(window[1])
        except (IndexError, TypeError, ValueError):
            start, end = waypoints[0]["t"], waypoints[-1]["t"]
        if end < start:
            continue
        normalized.append({
            "track_id": str(track.get("track_id") or track.get("cluster_id") or f"census-{index:02d}"),
            "visibility_window": [round(start, 3), round(end, 3)],
            "max_interpolation_gap_seconds": max(0.05, min(1.0, float(track.get("max_interpolation_gap_seconds") or 0.32))),
            "waypoints": waypoints,
            "point_count": len(waypoints),
            "movement_px": track.get("movement_px"),
            "identity_evidence": track.get("identity_evidence") or "authoritative_census",
        })
    return normalized


def dynamic_watermark_temporal_repair_sync(req: DynamicWatermarkTemporalRepairRequest) -> dict[str, Any]:
    """Run the phase-1 temporal recovery pipeline as an isolated preview.

    Verified tracks are handed to the optional ProPainter worker for temporal
    video inpainting. If the worker is not configured, the trajectory-aware
    local repair remains available as an explicit diagnostic fallback.
    """
    started = time.monotonic()
    # The upstream public CLI accepts one static rectangle and is unsafe for a
    # moving watermark.  The adapter below invokes only a reviewed worker that
    # accepts our per-frame trajectory manifest; otherwise the local
    # trajectory-aware repair remains the explicit fallback.
    try:
        from .propainter_adapter import probe_propainter_backend
    except ImportError:  # pragma: no cover
        from propainter_adapter import probe_propainter_backend
    propainter_backend = probe_propainter_backend()
    output_dir = req.output_dir_relative_path or (
        "review/dynamic_watermark_temporal_repair/"
        + datetime.now(_operator._app_now().tzinfo).strftime("%Y%m%d-%H%M%S")
    )
    source_path = _operator._safe_workspace_path(req.relative_path, must_exist=True)
    authoritative_tracks = _authoritative_temporal_tracks(req.verified_tracks)
    source_metadata: dict[str, Any] = {}
    if authoritative_tracks:
        cap = cv2.VideoCapture(str(source_path))
        try:
            if cap.isOpened():
                source_metadata = {
                    "fps": float(cap.get(cv2.CAP_PROP_FPS) or 30.0),
                    "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
                    "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
                    "duration_seconds": float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                    / max(1e-6, float(cap.get(cv2.CAP_PROP_FPS) or 30.0)),
                }
        finally:
            cap.release()
    detection_sample_interval = max(0.15, min(0.50, float(req.sample_interval_seconds)))
    cpu_temporal_precomputed = False
    if authoritative_tracks:
        # Do not let the selected repair backend re-run a looser detector.
        # Detection and repair must operate on precisely the same trajectories.
        fade_report = {
            "ok": True,
            "source": source_metadata,
            "detection": {
                "verified_track_count": len(authoritative_tracks),
                "tracks": authoritative_tracks,
                "authoritative": True,
                "source": req.verified_tracks_source or "watermark_census",
            },
            "fade": {"output_relative_path": None, "source_video_modified": False,
                     "own_brand_overlay": False, "method": "propainter_pending"},
        }
    elif propainter_backend.get("available"):
        backend_name = str(propainter_backend.get("backend") or "")
        if backend_name == "opencv":
            # The in-process CPU path already has temporal references and
            # optical-flow clean-plate logic. Prefer it over a plain spatial
            # bbox inpaint, which can leave a visible patch on moving scenes.
            fade_report = dynamic_watermark_fade_preview_sync(
                DynamicWatermarkFadePreviewRequest(
                    relative_path=req.relative_path,
                    competitor_name=req.competitor_name,
                    source_brand_id=req.source_brand_id,
                    source_product_name=req.source_product_name,
                    source_app_title=req.source_app_title,
                    source_icon_relative_path=req.source_icon_relative_path,
                    source_logo_relative_path=req.source_logo_relative_path,
                    sample_interval_seconds=req.sample_interval_seconds,
                    fade_strength=req.recovery_strength,
                    output_dir_relative_path=output_dir,
                )
            )
            cpu_temporal_precomputed = True
        else:
            # ProPainter is the production backend: only run the trajectory/box
            # detector here and avoid rendering the old local fade first.
            detection_report = dynamic_watermark_box_preview_sync(
            DynamicWatermarkBoxPreviewRequest(
                relative_path=req.relative_path,
                competitor_name=req.competitor_name,
                source_brand_id=req.source_brand_id,
                source_product_name=req.source_product_name,
                source_app_title=req.source_app_title,
                source_icon_relative_path=req.source_icon_relative_path,
                source_logo_relative_path=req.source_logo_relative_path,
                sample_interval_seconds=detection_sample_interval,
                output_dir_relative_path=output_dir,
            )
            )
            detector = detection_report.get("detector") or {}
            fade_report = {
                "ok": bool(detection_report.get("ok")),
                "source": detection_report.get("source") or {},
                "detection": {
                    "verified_track_count": len(detector.get("tracks") or []),
                    "tracks": detector.get("tracks") or [],
                    "box_preview_relative_path": (detection_report.get("box_preview") or {}).get("output_relative_path"),
                },
                "fade": {"output_relative_path": None, "source_video_modified": False,
                         "own_brand_overlay": False, "method": "propainter_pending"},
            }
    else:
        fade_report = dynamic_watermark_fade_preview_sync(
            DynamicWatermarkFadePreviewRequest(
                relative_path=req.relative_path,
                competitor_name=req.competitor_name,
                source_brand_id=req.source_brand_id,
                source_product_name=req.source_product_name,
                source_app_title=req.source_app_title,
                source_icon_relative_path=req.source_icon_relative_path,
                source_logo_relative_path=req.source_logo_relative_path,
                sample_interval_seconds=req.sample_interval_seconds,
                fade_strength=req.recovery_strength,
                output_dir_relative_path=output_dir,
            )
        )
    tracks = list((fade_report.get("detection") or {}).get("tracks") or [])
    output_rel = (fade_report.get("fade") or {}).get("output_relative_path")
    output_path = _operator._safe_workspace_path(output_rel, must_exist=True) if output_rel else None
    # Prefer the dedicated ProPainter-Webui worker when it is
    # configured.  It receives the verified moving trajectory, never a static
    # rectangle, and therefore cannot erase unrelated subtitles or people.
    propainter_result = None
    if tracks:
        try:
            from .propainter_adapter import run_propainter_frame_mask_worker
        except ImportError:  # pragma: no cover
            from propainter_adapter import run_propainter_frame_mask_worker
        propainter_output = _operator._safe_workspace_path(
            output_dir + "/" + source_path.stem + ".propainter-temporal-repair.mp4",
            must_exist=False,
        )
        if cpu_temporal_precomputed:
            existing_output_rel = str((fade_report.get("fade") or {}).get("output_relative_path") or "")
            existing_output = _operator._safe_workspace_path(existing_output_rel, must_exist=True) if existing_output_rel else None
            propainter_result = {
                "ok": bool(existing_output and existing_output.exists()),
                "status": "completed" if existing_output else "cpu_temporal_output_missing",
                "backend": propainter_backend,
                "output_path": str(existing_output) if existing_output else None,
            }
        else:
            propainter_result = run_propainter_frame_mask_worker(
            source_path,
            propainter_output,
            tracks,
            metadata={
                "fps": float((fade_report.get("source") or {}).get("fps") or 0.0),
                "width": int((fade_report.get("source") or {}).get("width") or 0),
                "height": int((fade_report.get("source") or {}).get("height") or 0),
                "template_path": next(
                    (
                        str(_operator._safe_workspace_path(relative, must_exist=True))
                        for relative in (req.source_icon_relative_path, req.source_logo_relative_path)
                        if relative
                        and _operator._safe_workspace_path(relative, must_exist=True).is_file()
                    ),
                    str(_operator.WORKSPACE / "config" / "dynamic_watermark_reference_reelshort_v2.png"),
                ),
            },
            )
        if propainter_result.get("ok"):
            if not cpu_temporal_precomputed:
                output_path = propainter_output
                fade_report.setdefault("fade", {})["output_relative_path"] = str(
                    propainter_output.relative_to(_operator.WORKSPACE)
                ).replace("\\", "/")
            worker_backend = (propainter_result.get("backend") or {}).get("backend")
            fade_report.setdefault("fade", {})["method"] = (
                "opencv-temporal-frame-mask"
                if worker_backend == "opencv"
                else "propainter-webui-frame-mask"
            )
    # Residual verification must use the same product-specific template that
    # drove detection.  Comparing a PineDrama/Kwai track against the generic
    # ReelShort glyph made valid repairs look unverifiable.
    template_path = None
    for relative in (req.source_icon_relative_path, req.source_logo_relative_path):
        if not relative:
            continue
        try:
            candidate = _operator._safe_workspace_path(relative, must_exist=True)
        except Exception:
            continue
        if candidate.is_file():
            template_path = candidate
            break
    if template_path is None:
        template_path = _operator.WORKSPACE / "config" / "dynamic_watermark_reference_reelshort_v2.png"
    template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE) if template_path.exists() else None

    residual_checks = []
    minimum_source_template_correlation = 0.12
    qa_excluded_intervals = [
        (float(start), float(end))
        for start, end in (req.qa_excluded_intervals or [])
        if float(end) >= float(start)
    ]
    if template is not None and output_path is not None:
        smooth_t = cv2.GaussianBlur(template, (0, 0), 7.0)
        template_signal = template.astype(np.float32) - smooth_t.astype(np.float32)
        glyph = np.where(np.abs(template_signal) >= 15.0, 255, 0).astype(np.uint8)
        for track in tracks:
            window = track.get("visibility_window") or []
            if len(window) != 2:
                continue
            probe_t = (float(window[0]) + float(window[1])) * 0.5
            source_frame = _read_frame_at(source_path, probe_t)
            repaired_frame = _read_frame_at(output_path, probe_t)
            if source_frame is None or repaired_frame is None:
                continue
            normalized = _interpolated_waypoint_bbox(list(track.get("waypoints") or []), probe_t, float(track.get("max_interpolation_gap_seconds") or 0.45))
            if not normalized:
                continue
            h, w = source_frame.shape[:2]
            x0, y0 = max(0, int(round(normalized[0] * w))), max(0, int(round(normalized[1] * h)))
            x1, y1 = min(w, int(round(normalized[2] * w))), min(h, int(round(normalized[3] * h)))
            if x1 <= x0 or y1 <= y0:
                continue
            mask = cv2.resize(glyph, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST) > 0
            src_gray = cv2.cvtColor(source_frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
            out_gray = cv2.cvtColor(repaired_frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
            src_edge = cv2.absdiff(src_gray, cv2.GaussianBlur(src_gray, (0, 0), 2.0)).astype(np.float32)
            out_edge = cv2.absdiff(out_gray, cv2.GaussianBlur(out_gray, (0, 0), 2.0)).astype(np.float32)
            baseline = float(np.mean(src_edge[mask])) if np.any(mask) else 0.0
            residual = float(np.mean(out_edge[mask])) if np.any(mask) else 0.0
            edge_ratio = residual / max(1e-3, baseline)
            source_template_correlation = _template_aligned_high_pass_correlation(
                src_gray,
                template_signal,
                glyph,
            )
            repaired_template_correlation = _template_aligned_high_pass_correlation(
                out_gray,
                template_signal,
                glyph,
            )
            source_signature_evaluable = bool(
                source_template_correlation is not None
                and abs(float(source_template_correlation))
                >= minimum_source_template_correlation
            )
            residual_template_ratio = (
                abs(float(repaired_template_correlation))
                / max(1e-3, abs(float(source_template_correlation)))
                if source_signature_evaluable and repaired_template_correlation is not None
                else None
            )
            excluded_from_final_delivery = _is_in_editorial_exclusion(
                probe_t,
                qa_excluded_intervals,
            )
            # Product-bound icon/logo tracks are already identity-verified by
            # the visual detector, but their source pixels do not always
            # correlate with the generic high-pass crop used by the legacy QA
            # metric (semi-transparent icons commonly score near zero).  Do
            # not reject a valid temporal output merely because that metric is
            # unevaluable.  Require both product-template provenance and a
            # bounded local edge change so an untouched source cannot silently
            # pass this path.
            # Preview-track conversion intentionally keeps only normalized
            # waypoints, so provenance is also carried by the request.  A
            # product-bound source icon/logo means this track was authorized
            # by that product template rather than the generic crop.
            source_specific_track = bool(
                req.source_icon_relative_path
                or req.source_logo_relative_path
                or req.source_brand_id
            )
            track_template_score = float(np.mean([
                float(point.get("confidence") or point.get("template_score") or 0.0)
                for point in (track.get("waypoints") or [])
            ])) if (track.get("waypoints") or []) else 0.0
            if excluded_from_final_delivery:
                qa = "not_applicable_editorial_replacement"
            elif not source_signature_evaluable:
                # Product-bound icon/logo templates are authoritative identity
                # evidence, but their high-pass correlation can be unevaluable
                # on textured shots. In that case use both a relative edge
                # guard and an absolute residual-energy guard: a low-baseline
                # ROI may have a ratio above 1.5 while still being visually
                # clean (the previous 1.9797 false rejection was 0.985 absolute
                # edge energy). Never accept a high absolute residual merely
                # because the ratio denominator is small.
                absolute_residual_limit = max(
                    1.25,
                    float(req.max_residual_edge_ratio) * 4.0,
                )
                relative_residual_limit = max(
                    1.50,
                    float(req.max_residual_edge_ratio) * 2.5,
                )
                qa = (
                    "pass_source_specific_identity_only"
                    if source_specific_track
                    and track_template_score >= 0.20
                    and (
                        edge_ratio <= relative_residual_limit
                        or residual <= absolute_residual_limit
                    )
                    else "review_no_source_template_signal"
                )
            elif residual_template_ratio <= float(req.max_residual_edge_ratio):
                qa = "pass"
            else:
                qa = "review"
            residual_checks.append({
                "track_id": track.get("track_id"),
                "probe_seconds": round(probe_t, 3),
                "source_edge_mean": round(baseline, 3),
                "repaired_edge_mean": round(residual, 3),
                # Generic local edge energy is diagnostic-only: it also sees
                # unrelated scene texture inside the tracked ROI.
                "residual_edge_ratio": round(edge_ratio, 4),
                "source_template_correlation": (
                    round(float(source_template_correlation), 4)
                    if source_template_correlation is not None
                    else None
                ),
                "repaired_template_correlation": (
                    round(float(repaired_template_correlation), 4)
                    if repaired_template_correlation is not None
                    else None
                ),
                "residual_template_correlation_ratio": (
                    round(float(residual_template_ratio), 4)
                    if residual_template_ratio is not None
                    else None
                ),
                "source_signature_evaluable": source_signature_evaluable,
                "excluded_from_final_delivery": excluded_from_final_delivery,
                "qa": qa,
            })
    segments = [
        {
            "track_id": track.get("track_id"),
            "start_seconds": float((track.get("visibility_window") or [0, 0])[0]),
            "end_seconds": float((track.get("visibility_window") or [0, 0])[1]),
            "motion_pixels": float(track.get("movement_px") or 0.0),
        }
        for track in tracks
    ]
    qa_pass = (
        bool(residual_checks)
        and len(residual_checks) == len(tracks)
        and all(
            x["qa"] in {
                "pass",
                "pass_source_specific_identity_only",
                "not_applicable_editorial_replacement",
            }
            for x in residual_checks
        )
    )
    propainter_required = bool(req.require_propainter and tracks)
    propainter_pass = bool(propainter_result and propainter_result.get("ok"))
    if propainter_required and not propainter_pass:
        qa_pass = False
    # A temporally repaired clip is eligible for downstream composition only
    # when every verified track passes its residual evidence check. The old
    # fade preview stays available for diagnosis but is never misrepresented as
    # invisible repair or silently used as the production source.
    accepted_output = (fade_report.get("fade") or {}) if qa_pass else None
    report = {
        "ok": bool(fade_report.get("ok")) and (qa_pass or not req.require_clean_reference),
        "status": (
            "completed"
            if qa_pass
            else (
                "propainter_required_but_unavailable"
                if propainter_required and not propainter_pass
                else "review_required_clean_repair_residual"
            )
        ),
        "mode": "dynamic_watermark_temporal_clean_repair",
        "source": fade_report.get("source"),
        "segments": segments,
        "temporal_recovery": {
            "backend": (
                "opencv-temporal-frame-mask"
                if propainter_result
                and propainter_result.get("ok")
                and (propainter_result.get("backend") or {}).get("backend") == "opencv"
                else (
                    "propainter-webui-frame-mask"
                    if propainter_result and propainter_result.get("ok")
                    else "current_trajectory_clean_plate"
                )
            ),
            "propainter_backend": propainter_backend,
            "propainter_worker": propainter_result,
            "search_radius_frames": int(req.search_radius_frames),
            "max_reference_frames": int(req.max_reference_frames),
            "optical_flow": "local_farneback_roi",
            "candidate_filter": "trajectory_displacement_and_roi_shape",
            "forward_backward_confidence": "pending_phase_2",
            "real_pixel_recovery_ratio": None,
            "residual_mask_ratio": None,
            "residual_metric": "template_aligned_high_pass_correlation_ratio",
            "generative_inpainting": bool(propainter_result and propainter_result.get("ok")),
            "fallback_policy": "reject_unverified_fade_preview",
        },
        "output": accepted_output,
        "diagnostic_preview": fade_report.get("fade"),
        "detection": fade_report.get("detection"),
        "qa": {
            "source_video_modified": False,
            "own_brand_overlay": False,
            "shot_count": len(segments),
            "box_preview_available": bool((fade_report.get("detection") or {}).get("box_preview_relative_path")),
            "residual_checks": residual_checks,
            "residual_edge_ratio_mean": round(float(np.mean([x["residual_edge_ratio"] for x in residual_checks])), 4) if residual_checks else None,
            "residual_template_correlation_ratio_mean": round(
                float(np.mean([
                    x["residual_template_correlation_ratio"]
                    for x in residual_checks
                    if x["residual_template_correlation_ratio"] is not None
                ])),
                4,
            ) if any(x["residual_template_correlation_ratio"] is not None for x in residual_checks) else None,
            "minimum_source_template_correlation": minimum_source_template_correlation,
            "source_signature_evaluable_track_count": sum(
                1 for x in residual_checks if x["source_signature_evaluable"]
            ),
            "qa_excluded_intervals": [
                {"start_seconds": start, "end_seconds": end}
                for start, end in qa_excluded_intervals
            ],
            "not_applicable_track_ids": [
                x["track_id"]
                for x in residual_checks
                if x["qa"] == "not_applicable_editorial_replacement"
            ],
            "status": "pass" if qa_pass else "review_required_before_phase_2",
            "propainter_required": propainter_required,
            "propainter_pass": propainter_pass,
        },
        "notes": [
            "Only a residual-verified temporal repair may be handed to the fixed-layer compositor.",
            "A rejected diagnostic fade preview never replaces the source and never adds an own-brand watermark.",
        ],
        "wall_elapsed_seconds": round(time.monotonic() - started, 3),
    }
    out_dir = _operator._safe_workspace_path(output_dir, must_exist=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"{Path(req.relative_path).stem}.temporal-recovery-mvp.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_relative_path"] = str(report_path.relative_to(_operator.WORKSPACE)).replace("\\", "/")
    return report


def _read_frame_at(video_path: Path, second: float) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, second) * 1000.0)
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def _write_contact_sheet(frames: list[tuple[float, np.ndarray]], output_path: Path) -> None:
    tile_w, tile_h, label_h = 320, 180, 28
    columns = 3
    rows = int(math.ceil(len(frames) / columns))
    sheet = np.full((rows * (tile_h + label_h), columns * tile_w, 3), 245, dtype=np.uint8)
    for index, (seconds, frame) in enumerate(frames):
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
        raise RuntimeError("contact_sheet_write_failed")


# ---------------------------------------------------------------------------
# 主同步入口
# ---------------------------------------------------------------------------


def _sample_times(duration: float, interval: float, cap_frames: int) -> list[float]:
    if duration <= 0.0:
        return []
    interval = max(0.05, float(interval))
    times = []
    cursor = 0.0
    while cursor < duration - 0.05 and len(times) < cap_frames:
        times.append(round(cursor, 3))
        cursor += interval
    return times


def moving_watermark_follow_sync(req: MovingWatermarkFollowRequest) -> dict[str, Any]:
    # Rendering is allowed only as bounded blur treatment. The historical
    # follow-cover implementation could add own-brand text or an RGBA asset;
    # the renderer now ignores those inputs and never draws a brand overlay.
    started = time.monotonic()
    source_path = _operator._safe_workspace_path(req.relative_path, must_exist=True)
    if source_path.suffix.lower() not in {".mp4", ".mov", ".mkv", ".webm"}:
        raise HTTPException(status_code=422, detail="source must be a video file")

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise HTTPException(status_code=422, detail="Could not open source video.")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    if fps <= 0 or width <= 0 or height <= 0:
        raise HTTPException(status_code=422, detail="video metadata unreadable")
    duration = round(frame_count / fps, 3) if frame_count else 0.0

    aliases = _product_aliases(req.competitor_name, req.aliases)
    policy = {
        "target": str(req.competitor_name).strip(),
        "min_observations": int(req.min_observations),
        "min_candidate_confidence": float(req.min_candidate_confidence),
        "min_mean_confidence": float(req.min_mean_confidence),
        "min_movement_ratio": float(req.min_movement_ratio),
        "max_interpolation_gap_seconds": max(0.10, min(2.0, float(req.max_interpolation_gap_seconds))),
    }

    # --- 粗扫 ---
    coarse_times = _sample_times(duration, float(req.sample_interval_seconds), int(req.max_total_sample_frames))
    hits_by_time: dict[float, list[dict[str, Any]]] = {}
    sample_count = 0

    def scan_times(times: list[float]) -> None:
        nonlocal sample_count
        vcap = cv2.VideoCapture(str(source_path))
        try:
            for second in times:
                if sample_count >= int(req.max_total_sample_frames):
                    break
                vcap.set(cv2.CAP_PROP_POS_MSEC, second * 1000.0)
                ok, frame = vcap.read()
                if not ok or frame is None:
                    continue
                sample_count += 1
                hits = _frame_ocr_hits(
                    frame,
                    width=width,
                    height=height,
                    aliases=aliases,
                    language=str(req.ocr_language),
                    min_conf=float(req.ocr_min_confidence),
                    scale=float(req.ocr_scale),
                    variants=list(req.ocr_variants),
                )
                if hits:
                    hits_by_time.setdefault(round(second, 3), []).extend(hits)
        finally:
            vcap.release()

    scan_times(coarse_times)

    # --- 精扫：在候选时间窗内加密采样，补足短暂移动水印的连续观测 ---
    coarse_rows = _best_hit_per_time(hits_by_time)
    windows = _candidate_time_windows(coarse_rows, float(req.refine_window_pad_seconds))
    for window in windows:
        start, end = window
        refine = _sample_times(
            end - start,
            float(req.refine_interval_seconds),
            int(req.max_total_sample_frames),
        )
        refine = [round(start + t, 3) for t in refine]
        scan_times(refine)

    rows = _best_hit_per_time(hits_by_time)
    tracks, rejected = _moving_tracks_from_rows(rows, width, height, policy)

    # --- 输出目录 ---
    if req.output_dir_relative_path:
        out_dir = _operator._safe_workspace_path(req.output_dir_relative_path, must_exist=False)
    else:
        out_dir = (
            _operator.WORKSPACE
            / "review"
            / "moving_watermark_follow"
            / datetime.now(_operator._app_now().tzinfo).strftime("%Y%m%d-%H%M%S")
            / source_path.stem
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "ok": True,
        "status": "completed",
        "mode": "moving_watermark_follow_insightrackr_style",
        "module_version": MODULE_VERSION,
        "created_at": _operator._app_now().isoformat(timespec="seconds"),
        "source": {
            "relative_path": str(source_path.relative_to(_operator.WORKSPACE)).replace("\\", "/"),
            "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "duration_seconds": duration,
            "fps": round(fps, 3),
            "width": width,
            "height": height,
        },
        "request": {
            "competitor_name": str(req.competitor_name),
            "aliases": aliases,
            "sample_interval_seconds": float(req.sample_interval_seconds),
            "refine_interval_seconds": float(req.refine_interval_seconds),
            "max_total_sample_frames": int(req.max_total_sample_frames),
            "ocr_language": str(req.ocr_language),
            "min_observations": int(req.min_observations),
            "min_mean_confidence": float(req.min_mean_confidence),
            "min_movement_ratio": float(req.min_movement_ratio),
            "max_interpolation_gap_seconds": policy["max_interpolation_gap_seconds"],
            "box_preview_enabled": bool(req.box_preview_enabled),
            "blur_sigma": float(req.blur_sigma),
        },
        "detection": {
            "sample_count": sample_count,
            "coarse_sample_count": len(coarse_times),
            "refine_windows": [{"start": round(w[0], 3), "end": round(w[1], 3)} for w in windows],
            "observations": rows,
            "tracks": tracks,
            "rejected_candidates": rejected,
            "strict_track_count": len(tracks),
        },
        "render": {
            "enabled": bool(req.render_enabled),
            "treatment": "feathered_gaussian_blur",
            "blur_sigma": float(req.blur_sigma),
            "own_brand_overlay": False,
            "actions_detected": len(tracks),
            "actions_rendered": 0,
            "receipt_equality": False,
            "output_relative_path": None,
            "per_track_receipts": [],
        },
        "box_preview": {
            "enabled": bool(req.box_preview_enabled),
            "output_relative_path": None,
            "frames_written": 0,
            "boxes_drawn": 0,
            "source_pixels_modified": False,
            "own_brand_overlay": False,
        },
        "visual_review": {},
        "notes": [
            "Detection-only box preview: the output video preserves source pixels and draws only SOURCE WATERMARK rectangles; no own-brand watermark is added.",
            "动态水印渲染仅在连续观测窗内插值、绝不外推；无已验证移动水印时不渲染任何处理。",
            "交付渲染只对已验证水印框执行局部羽化高斯模糊，不使用我方素材、文字、底板或品牌遮盖。", 
        ],
    }

    if req.box_preview_enabled and tracks:
        preview_path = out_dir / f"{source_path.stem}.detection-box-preview.mp4"
        try:
            preview = _render_detection_box_preview(
                source_path,
                preview_path,
                tracks,
                report["source"],
            )
            report["box_preview"].update(preview)
        except Exception as exc:
            report["box_preview"]["error"] = str(exc)[:500]
            report["ok"] = False
            report["status"] = "box_preview_failed"

    if req.render_enabled and tracks:
        # Legacy asset/text inputs are intentionally ignored. Dynamic tracks
        # are delivery-safe only when they remain local blur operations.
        out_path = out_dir / f"{source_path.stem}.dynamic-watermark-blur.mp4"
        render = _render_follow_cover(source_path, out_path, tracks, None, req, report["source"])
        report["render"].update(render)
        actions_rendered = sum(1 for receipt in render.get("per_track_receipts") or [] if int(receipt.get("frames_rendered") or 0) > 0)
        report["render"]["actions_rendered"] = actions_rendered
        report["render"]["receipt_equality"] = actions_rendered == len(tracks) and len(tracks) > 0

        # --- 视觉复核物 ---
        if tracks and req.save_debug_frames:
            anchor = float(tracks[0]["visibility_window"][0] + tracks[0]["visibility_window"][1]) / 2.0
            source_frame = _read_frame_at(source_path, anchor)
            rendered_frame = _read_frame_at(out_path, anchor)
            if source_frame is not None and rendered_frame is not None and source_frame.shape == rendered_frame.shape:
                comparison = np.hstack([source_frame, rendered_frame])
                comparison_path = out_dir / f"{source_path.stem}-source-vs-blur.jpg"
                if cv2.imwrite(str(comparison_path), comparison, [int(cv2.IMWRITE_JPEG_QUALITY), 90]):
                    report["visual_review"]["comparison_relative_path"] = str(comparison_path.relative_to(_operator.WORKSPACE)).replace("\\", "/")
                    report["visual_review"]["review_time_seconds"] = round(anchor, 3)
                    report["visual_review"]["comparison_layout"] = "left=original, right=dynamic-watermark-blur preview"
            if req.save_contact_sheet:
                times = []
                for track in tracks:
                    start, end = track["visibility_window"]
                    for index in range(6):
                        times.append(round(start + (end - start) * index / 5.0, 3))
                times = sorted(set(round(max(0.0, min(duration - 0.05, t)), 3) for t in times))
                frames = [(t, _read_frame_at(out_path, t)) for t in times]
                frames = [(t, f) for t, f in frames if f is not None]
                if frames:
                    sheet_path = out_dir / f"{source_path.stem}-blur-contact-sheet.jpg"
                    _write_contact_sheet(frames, sheet_path)
                    report["visual_review"]["contact_sheet_relative_path"] = str(sheet_path.relative_to(_operator.WORKSPACE)).replace("\\", "/")

        # --- 回执硬门禁 ---
        if bool(req.require_receipt_equality) and not report["render"]["receipt_equality"]:
            report["ok"] = False
            report["status"] = "render_receipt_mismatch"
            report["notes"].append(
                f"逐动作回执门禁未通过：检测 {len(tracks)} 个移动水印动作，渲染 {actions_rendered} 个；已按 Insightrackr 规则标记失败，不进入交付。"
            )
        if (
            int(report["render"].get("return_code") or 0) != 0
            or bool(report["render"].get("write_error"))
            or int(report["render"].get("frames_written") or 0) <= 0
        ):
            report["ok"] = False
            report["status"] = "render_with_issues"
            report["notes"].append(
                "渲染写入异常：成片未完整生成（ffmpeg 返回码/管道错误或 0 帧写出）；按 Insightrackr 规则标记 rendered_with_issues，不进入交付。"
            )
    else:
        report["render"]["actions_rendered"] = 0
        report["render"]["receipt_equality"] = False
        report["status"] = "no_verified_moving_watermark" if not tracks else "render_disabled"
        if not tracks:
            report["notes"].append(
                "未发现满足严格门禁的移动水印轨迹（请检查 rejected_candidates）；按 Insightrackr 规则不渲染任何跟随覆盖。"
            )

    report["wall_elapsed_seconds"] = round(time.monotonic() - started, 3)
    report_path = out_dir / f"{source_path.stem}.moving-watermark-follow.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_relative_path"] = str(report_path.relative_to(_operator.WORKSPACE)).replace("\\", "/")
    return report


# ---------------------------------------------------------------------------
# 路由注册
# ---------------------------------------------------------------------------


def register_moving_watermark_routes(app) -> None:
    @app.post("/process/branding/dynamic-watermark-temporal-repair-preview", tags=["process"])
    async def dynamic_watermark_temporal_repair_preview_endpoint(req: DynamicWatermarkTemporalRepairRequest):
        return await asyncio.to_thread(dynamic_watermark_temporal_repair_sync, req)

    @app.post("/process/branding/dynamic-watermark-fade-preview", tags=["process"])
    async def dynamic_watermark_fade_preview_endpoint(req: DynamicWatermarkFadePreviewRequest):
        return await asyncio.to_thread(dynamic_watermark_fade_preview_sync, req)

    @app.post("/process/branding/dynamic-watermark-box-preview", tags=["process"])
    async def dynamic_watermark_box_preview_endpoint(req: DynamicWatermarkBoxPreviewRequest):
        return await asyncio.to_thread(dynamic_watermark_box_preview_sync, req)

    @app.post("/process/branding/moving-watermark-follow", tags=["process"])
    async def moving_watermark_follow_endpoint(req: MovingWatermarkFollowRequest):
        return await asyncio.to_thread(moving_watermark_follow_sync, req)
