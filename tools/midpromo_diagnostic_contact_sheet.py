"""Create a read-only 45–67s diagnostic contact sheet for a Creative Loop source video."""

from __future__ import annotations

import json
from pathlib import Path
import os

import cv2
import numpy as np


SOURCE = Path(
    os.getenv(
        "SOURCE_VIDEO",
        r"D:\creative-loop-floatboat\workspace\raw\2026-08-16_0512\ReelShort\Portuguese\x_v_06301722f6dafcbc9b23b864110b732e.mp4",
    )
)
OUTPUT = SOURCE.parent / "_debug_midpromo_45_67.jpg"
SAMPLE_SECONDS = [44.5, 45, 45.5, 46, 50, 55, 60, 65, 66.5, 67, 67.5]


def main() -> None:
    cap = cv2.VideoCapture(str(SOURCE))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {SOURCE}")

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        thumbs: list[np.ndarray] = []

        for seconds in SAMPLE_SECONDS:
            cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            frame = cv2.resize(frame, (180, 320))
            text = f"{seconds:.1f}s"
            cv2.putText(frame, text, (7, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, text, (7, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
            thumbs.append(frame)
    finally:
        cap.release()

    if not thumbs:
        raise RuntimeError("No diagnostic frames extracted")

    rows: list[np.ndarray] = []
    for index in range(0, len(thumbs), 6):
        row = thumbs[index : index + 6]
        while len(row) < 6:
            row.append(np.zeros_like(thumbs[0]))
        rows.append(np.hstack(row))

    if not cv2.imwrite(str(OUTPUT), np.vstack(rows)):
        raise RuntimeError(f"Could not write {OUTPUT}")

    print(
        json.dumps(
            {
                "source": str(SOURCE),
                "output": str(OUTPUT),
                "fps": fps,
                "frame_count": frame_count,
                "duration_seconds": round(frame_count / fps, 3),
                "sample_seconds": SAMPLE_SECONDS,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()