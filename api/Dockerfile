FROM mcr.microsoft.com/playwright/python:v1.61.0-noble AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl tesseract-ocr tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

# The Engine executes long-running render and upload jobs.  Development reload
# can terminate the in-process worker thread mid-job, so production starts a
# stable Uvicorn process. Rebuild with creative-loop-control.cmd rebuild after
# changing application code.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# CPU image target used by the default compose file.
FROM runtime AS base
COPY app ./app

# Optional image target used only by compose.gpu.yaml. Keeping it separate
# avoids forcing the CUDA runtime onto normal CPU-only deployments.
FROM runtime AS gpu-dependencies
# RTX 50-series (Blackwell, sm_120) requires a PyTorch CUDA 12.8-or-newer
# build.  cu121 wheels can enumerate the device but cannot execute its kernels.
# Torch/TorchVision are installed from local wheels under api/wheels/ so the
# build does not depend on the slow public PyTorch index (download.pytorch.org
# or the flat Aliyun mirror listing, which is not a pip PEP 503 index).
COPY wheels /wheels
RUN pip install --no-cache-dir \
        /wheels/torch-2.7.1+cu128-cp312-cp312-manylinux_2_28_x86_64.whl \
        /wheels/torchvision-0.22.1+cu128-cp312-cp312-manylinux_2_28_x86_64.whl \
    && pip install --no-cache-dir \
        addict av easydict einops future imageio imageio-ffmpeg matplotlib \
        opencv-python pyyaml requests scikit-image timm tqdm yapf

# Keep CUDA/PyTorch below application code in the layer graph. App-only changes
# should then reuse this expensive layer instead of downloading GPU packages.
FROM gpu-dependencies AS gpu
COPY app ./app

# Creative Loop branding OCR runtime is installed in the single apt layer above.
