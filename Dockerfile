# syntax=docker/dockerfile:1
# Garment attribute API, CPU only. Fashion-CLIP (~600 MB) is mounted read-only from a
# Hugging Face cache folder at run time (docker-compose.yml), not baked into a layer.
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.11.17 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY app/ app/
COPY services/ services/
COPY schemas/ schemas/
COPY demo/ demo/
COPY run.py ./

RUN useradd --uid 1000 --create-home garment
USER garment

ENV PATH=/app/.venv/bin:$PATH \
    GARMENT_MODELS_DIR=/models \
    HF_HUB_OFFLINE=1
EXPOSE 8003
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8003/health')" || exit 1
CMD ["python", "run.py", "--host", "0.0.0.0", "--port", "8003"]
