# Creative Loop API Starter

Architecture:

Floatboat -> HTTP API -> Docker/Python -> workspace

## Hardware requirements for dynamic watermark repair

The project can run without a GPU, but the repair backend depends on the available hardware:

| Target | Minimum guidance | Backend |
| --- | --- | --- |
| CPU/OpenCV | No dedicated GPU, or less than 6 GB VRAM; 8-core CPU and 16 GB RAM recommended | `opencv` / `cpu` |
| CUDA entry level | NVIDIA GTX 1660/1660 Super 6 GB, RTX 2060 6 GB, RTX 3050 8 GB, or newer equivalent | `cuda` + ProPainter |
| CUDA recommended | 8 GB VRAM for 720p; 12 GB or more for 1080p/long videos, for example RTX 3060 12 GB or RTX 4070 12 GB | `cuda` + ProPainter |
| AMD/Intel alternative | Linux ROCm or DirectML-compatible GPU, preferably 8 GB or more | `rocm`, `directml`, or OpenCV |

Cards such as GeForce MX250 2 GB are below the CUDA/ProPainter recommendation. They may run the CPU/OpenCV path, but must not be treated as a validated CUDA/ProPainter machine. Actual limits also depend on input resolution, sub-video length, model weights, and concurrent jobs.

After deployment, verify the selected backend inside the API container:

```powershell
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
docker compose exec creative-loop-api python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-cuda')"
docker compose exec creative-loop-api python -c "from app.propainter_adapter import probe_propainter_backend; print(probe_propainter_backend())"
```

`propainter_cuda_configured` means the CUDA/ProPainter path is configured. `opencv_temporal_mask_configured` means the service is using the CPU/OpenCV fallback; a visible NVIDIA card alone does not change that status.

### NVIDIA / ProPainter container deployment

The normal `compose.yaml` intentionally starts on every machine and does not expose a GPU. On a configured NVIDIA host, copy `.env.example` to `.env`, set `PROPAINTER_HOST_ROOT` to the directory containing `ProPainter/inference_propainter.py` and the model weights, then start the GPU override. This override builds the separate CUDA-enabled API image; the normal image cannot use ProPainter merely because the host has a GPU.

```powershell
docker compose -f compose.yaml -f compose.gpu.yaml up -d --build
docker compose -f compose.yaml -f compose.gpu.yaml exec creative-loop-api python -c "from app.propainter_adapter import probe_propainter_backend; print(probe_propainter_backend())"
```

If the second command does not report `propainter_cuda_configured`, the service will use the explicitly-labelled OpenCV fallback rather than claiming ProPainter has run.

## 1. Start

PowerShell:

```powershell
cd C:\creative-loop
docker compose up -d --build
docker compose ps
```

## 2. Health check

```powershell
Invoke-RestMethod http://localhost:8000/health
```

## 3. Echo test

```powershell
$body = @{ message = "hello from Floatboat" } | ConvertTo-Json
Invoke-RestMethod -Method Post `
  -Uri http://localhost:8000/tools/echo `
  -ContentType "application/json" `
  -Body $body
```

## 4. Browser probe

```powershell
$body = @{ url = "https://example.com" } | ConvertTo-Json
Invoke-RestMethod -Method Post `
  -Uri http://localhost:8000/browser/probe `
  -ContentType "application/json" `
  -Body $body
```

## 5. Multi-URL crawl scaffold

```powershell
$body = @{
  urls = @(
    "https://example.com",
    "https://example.org"
  )
} | ConvertTo-Json

Invoke-RestMethod -Method Post `
  -Uri http://localhost:8000/crawl/url `
  -ContentType "application/json" `
  -Body $body
```

## 6. API docs

Open:

http://localhost:8000/docs

## Folder roles

- `workspace/raw` original downloaded materials
- `workspace/processed` automatically processed materials
- `workspace/review` low-confidence/manual-review materials
- `workspace/output` final upload-ready materials
- `browser-state` reserved for browser login/session state
- `logs` service logs

## Stop

```powershell
docker compose down
```
