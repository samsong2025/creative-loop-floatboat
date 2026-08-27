"""Report the actual dynamic-watermark backend selected by this runtime.

Run from a development environment with the API dependencies installed, or
use this equivalent command inside the API container:
    docker compose -f compose.yaml -f compose.gpu.yaml exec creative-loop-api \
      python -c "from app.propainter_adapter import probe_propainter_backend; print(probe_propainter_backend())"
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))

from app.propainter_adapter import probe_propainter_backend


def main() -> int:
    report = {"propainter": probe_propainter_backend()}
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=False,
        )
        report["nvidia_smi"] = result.stdout.strip() if result.returncode == 0 else result.stderr.strip()
    else:
        report["nvidia_smi"] = "not_available_in_container"
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["propainter"].get("status") == "propainter_cuda_configured" else 2


if __name__ == "__main__":
    raise SystemExit(main())
