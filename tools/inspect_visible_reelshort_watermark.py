"""Inspect visible ReelShort watermark locations across a source or rendered video.

This diagnostic is intentionally read-only: it emits OCR observations and a
contact sheet so a rendering fix can be validated against visible pixels.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "api"))

from app import branding_v09 as branding  # noqa: E402


def inspect(video_path: Path, seconds: list[float]) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    try:
        for second in seconds:
            cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                print(f"{second:05.2f}s unreadable")
                continue
            observations = []
            for variant in ("original", "clahe", "tophat", "blackhat"):
                try:
                    image = dict(branding._watermark_preprocess_variants(frame, [variant]))[variant]
                    words = branding._tesseract_words(image, "eng", 10.0, 2.25)
                except Exception as exc:
                    observations.append({"variant": variant, "error": str(exc)[:120]})
                    continue
                for word in words:
                    text = str(word.get("text") or "").strip()
                    compact = "".join(char for char in text.casefold() if char.isalnum())
                    if "reel" in compact or "short" in compact:
                        observations.append(
                            {
                                "variant": variant,
                                "text": text,
                                "confidence": round(float(word.get("conf") or 0.0), 2),
                                "bbox": [
                                    int(word.get("x") or 0),
                                    int(word.get("y") or 0),
                                    int(word.get("width") or 0),
                                    int(word.get("height") or 0),
                                ],
                            }
                        )
            print(f"{second:05.2f}s {observations}")
    finally:
        cap.release()


if __name__ == "__main__":
    inspect(
        Path("/workspace/raw/2026-08-16_0512/ReelShort/English/x_v_259079a89647cd48c07e492b716738a1.mp4"),
        [round(value * 0.5, 2) for value in range(0, 57)],
    )