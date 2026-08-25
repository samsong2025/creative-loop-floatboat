from __future__ import annotations

from typing import Any


def build_residual_report(analysis: dict[str, Any], *, threshold: float = 0.80) -> dict[str, Any]:
    """Turn a post-render visual scan into a bounded local-repair gate.

    v1 may identify a translucent logo primarily through edge matching, so QC
    uses the strongest visual evidence rather than the old icon-only score.
    ``bbox`` is preserved in each interval to let the repair path constrain its
    re-render scope to the missed screen-space region.
    """
    def confidence(item: dict[str, Any]) -> float:
        return max(
            float(item.get("icon_similarity", 0.0)),
            float(item.get("edge_similarity", 0.0)),
            float(item.get("detector_confidence", 0.0)),
            float(item.get("score", 0.0)),
        )

    candidates = [item for item in analysis.get("candidates", []) if confidence(item) >= threshold]
    intervals: list[dict[str, Any]] = []
    for item in candidates:
        start = float(item.get("time_seconds", 0.0))
        if intervals and start - intervals[-1]["end_seconds"] <= 1.5:
            intervals[-1]["end_seconds"] = start
            intervals[-1]["max_confidence"] = max(intervals[-1]["max_confidence"], confidence(item))
            if len(intervals[-1]["bbox_samples"]) < 12 and item.get("bbox"):
                intervals[-1]["bbox_samples"].append(item["bbox"])
        else:
            intervals.append(
                {
                    "start_seconds": start,
                    "end_seconds": start,
                    "max_confidence": confidence(item),
                    "evidence": item.get("source", "visual_match"),
                    "bbox_samples": [item["bbox"]] if item.get("bbox") else [],
                }
            )
    for interval in intervals:
        # Preserve the previous lean payload for callers without geometry, but
        # omit an empty field so their schema validation remains compatible.
        if not interval["bbox_samples"]:
            interval.pop("bbox_samples")
    return {
        "brand_id": analysis.get("brand_id"),
        "pass": not intervals,
        "max_confidence": max((confidence(item) for item in candidates), default=0.0),
        "intervals": intervals,
        "metrics": {
            "sampled_frames": analysis.get("metrics", {}).get("sampled_frames", 0),
            "residual_scan_runtime_ms": analysis.get("metrics", {}).get("detector_runtime_ms", 0),
            "residual_policy": "visual_icon_or_edge_confidence__local_bbox_repair",
        },
    }