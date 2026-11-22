# Container for the web app: `docker build -t chesshacks . && docker run -p 8000:8000 chesshacks`
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    TORCH_THREADS=2

WORKDIR /app

# CPU-only PyTorch keeps the image a few hundred MB instead of several GB.
# Override TORCH_INDEX to build against a different wheel index (e.g. CUDA).
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --index-url ${TORCH_INDEX} torch \
 && pip install -r requirements.txt

COPY src ./src
COPY app ./app
COPY scripts ./scripts

# If the build context only has Git LFS pointers, download the real weights.
RUN python scripts/fetch_weights.py

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.server:app --host 0.0.0.0 --port ${PORT}"]
