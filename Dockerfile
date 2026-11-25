# API and worker image (same image, different command). No torch: MiniFASNet isn't part of this flow.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PATH="/app/.venv/bin:$PATH"
RUN pip install --no-cache-dir uv

WORKDIR /app
# dependencies first, so code changes don't reinstall them
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra api --no-install-project
COPY src ./src
COPY resources/face_detection ./resources/face_detection
RUN uv sync --frozen --no-dev --extra api

RUN useradd --uid 10001 --no-create-home app && mkdir -p /app/var && chown app /app/var
USER app
EXPOSE 8000
CMD ["uvicorn", "liveness_service.api.http.app:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
