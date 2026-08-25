"""Build a read-only contact sheet proving the user-confirmed 45–67s range."""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path


SOURCE = Path("/workspace/raw/2026-08-16_0512/ReelShort/Portuguese/x_v_06301722f6dafcbc9b23b864110b732e.mp4")
OUTPUT = Path("/workspace/review/_midpromo_render_coverage.jpg")
TIMES = [44.7, 44.9, 45.0, 45.1, 46.0, 50.0, 55.0, 60.0, 65.0, 67.0, 67.2, 67.4]


def main() -> None:
    cap = cv2.VideoCapture(str(SOURCE))
    if not cap.isOpened():
        raise RuntimeError("Could not open source video")
    thumbs: list[np.ndarray] = []
    try:
        for seconds in TIMES:
            cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"Could not read {seconds}s")
            thumbnail = cv2.resize(frame, (180, 320))
            x = 7 + (len(thumbs) % 6) * 180
            y = 25 + (len(thumbs) // 6) * 320
            cv2.putText(thumbnail, f"{seconds:.1f}s", (7, 25), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(thumbnail, f"{seconds:.1f}s", (7, 25), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
            thumbs.append(thumbnail)
    finally:
        cap.release()
    rows = [np.hstack(thumbs[index : index + 6]) for index in range(0, len(thumbs), 6)]
    if not cv2.imwrite(str(OUTPUT), np.vstack(rows)):
        raise RuntimeError("Could not write coverage sheet")
    print(OUTPUT)


if __name__ == "__main__":
    main()