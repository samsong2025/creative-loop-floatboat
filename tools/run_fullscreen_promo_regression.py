"""Run offline regression for generic full-screen mid-promo detection.

Read-only: this script imports the existing detector, analyses fixture videos,
and writes one JSON report under workspace/review/regression. It never renders,
changes source media, starts a task, approves, or uploads.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path


WORKSPACE = Path("/workspace")
sys.path.insert(0, "/app")
from app import branding_v09 as branding  # noqa: E402


def within(value: float, bounds: list[float]) -> bool:
    return float(bounds[0]) <= float(value) <= float(bounds[1])


def main() -> None:
    config_path = WORKSPACE / "config" / "fullscreen_promo_regression_cases.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    results = []

    for case in config.get("cases") or []:
        relative_path = str(case["source_relative_path"])
        video_path = WORKSPACE / relative_path
        media = branding._probe_media(video_path)
        semantic = None
        semantic_relative_path = case.get("semantic_report_relative_path")
        if semantic_relative_path:
            semantic = json.loads((WORKSPACE / str(semantic_relative_path)).read_text(encoding="utf-8"))
        started = __import__("time").monotonic()
        scan = branding._fullscreen_promo_scan(
            video_path,
            float(media.get("duration_seconds") or 0.0),
            sample_interval_seconds=0.50,
            semantic_report=semantic,
        )
        candidates = scan.get("candidates") or []
        result = {
            "case_id": case["case_id"],
            "kind": case["kind"],
            "source_relative_path": relative_path,
            "candidate_count": len(candidates),
            "elapsed_seconds": round(__import__("time").monotonic() - started, 3),
            "top_candidate": candidates[0] if candidates else None,
            "pass": False,
        }
        if case["kind"] == "positive":
            top = candidates[0] if candidates else None
            result["pass"] = bool(
                top
                and within(float(top["start_seconds"]), case["expected_start_range_seconds"])
                and within(float(top["end_seconds"]), case["expected_end_range_seconds"])
                and (
                    not case.get("expected_boundary_source")
                    or str((top.get("after_cut") or {}).get("source") or "")
                    == str(case["expected_boundary_source"])
                )
            )
        else:
            result["pass"] = len(candidates) == int(case["expected_candidate_count"])
        results.append(result)

    passed = all(item["pass"] for item in results)
    report = {
        "ok": passed,
        "mode": "fullscreen_promo_detection_regression",
        "created_at": datetime.now().astimezone().isoformat(),
        "config_relative_path": "config/fullscreen_promo_regression_cases.json",
        "results": results,
    }
    out_dir = WORKSPACE / "review" / "regression" / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fullscreen-promo-regression.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok": passed, "report": str(out_path), "results": results}, ensure_ascii=False))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()