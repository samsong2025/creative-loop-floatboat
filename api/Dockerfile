FROM mcr.microsoft.com/playwright/python:v1.61.0-noble

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl tesseract-ocr tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8000

# The Engine executes long-running render and upload jobs.  Development reload
# can terminate the in-process worker thread mid-job, so production starts a
# stable Uvicorn process. Rebuild with creative-loop-control.cmd rebuild after
# changing application code.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Creative Loop branding OCR runtime is installed in the single apt layer above.
