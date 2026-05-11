FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt .
# CPU-only torch. Inference runs on CPU in Docker.
RUN pip install --no-cache-dir --timeout 120 \
    torch==2.6.0+cpu torchvision==0.21.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --timeout 120 -r requirements-docker.txt

COPY cbr_news ./cbr_news
COPY api ./api
COPY bot ./bot
COPY configs ./configs

ENV PYTHONPATH=/app
ENV CHECKPOINT_PATH=/app/checkpoints

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
