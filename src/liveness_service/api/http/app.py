"""HTTP API for the mobile app: video liveness under /v1/liveness.

Flow for the app (every request carries X-API-Key):
  1. tus create: POST /v1/liveness. The Location header is /v1/liveness/{upload_id}.
  2. GET /v1/liveness/{upload_id}/events (SSE) and keep it open.
  3. tus PATCH /v1/liveness/{upload_id} chunks; HEAD it to get the offset to resume from.
     The stream sends `progress`, then `processing`, then one final `result` or `error`, and closes.
     Reconnect with Last-Event-ID to resume the stream.
Uploads are forwarded to tusd, which calls /tusd/hooks; after post-finish the video is checked.

Run: TUSD_URL=http://localhost:8021/files/ TUSD_DATA_DIR=./data \
     uv run uvicorn liveness_service.api.http.app:app --host 0.0.0.0 --port 8000
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from liveness_service.api.http.routes import events, health, tusd_hooks, uploads
from liveness_service.core import tusd_client
from liveness_service.ml.checker import LivenessChecker


@asynccontextmanager
async def lifespan(app):
    app.state.checker = LivenessChecker()  # load models once per worker, not per request
    app.state.tasks = set()                # running video checks
    app.state.tusd = tusd_client.create_client()
    yield
    await app.state.tusd.aclose()


app = FastAPI(title="Liveness Service", lifespan=lifespan)
for module in (health, uploads, events, tusd_hooks):
    app.include_router(module.router)
