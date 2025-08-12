# ---------- UI build stage ----------
FROM node:20-alpine AS ui-builder
WORKDIR /app/ui

# If you use pnpm or yarn, adjust below
COPY ui/package*.json ./
RUN npm ci

COPY ui/ ./
RUN npm run build

# ---------- Python runtime ----------
FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ONLY_BINARY=:all: \
    PYTHONUNBUFFERED=1 \
    PORT=7860

# tools that some wheels rely on; tiny
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
# +cpu index gives you a small CPU torch wheel; drop if you bring your own CUDA base
RUN python -m pip install --upgrade pip && \
    pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# copy app after deps (better layer caching)
COPY backend/ ./backend/
COPY ui/dist ./ui/dist

EXPOSE 7860
HEALTHCHECK CMD curl -f http://localhost:${PORT}/health || exit 1
CMD ["uvicorn", "backend.api:api", "--host", "0.0.0.0", "--port", "7860"]
