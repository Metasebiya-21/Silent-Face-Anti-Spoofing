"""HTTP API for the mobile app: liveness sessions with a colour challenge, under /v1/liveness.

Flow for the app:
  1. POST /v1/liveness/sessions (X-API-Key)                  -> session_id, token, tus_endpoint, expires_at
  2. GET  /v1/liveness/sessions/{id}/events (Bearer token)   -> SSE: `capture` (start recording), then each
     `colour` just in time: show it immediately and note when it appeared on screen.
  3. POST /v1/liveness/sessions/{id}/display-log              -> when each colour was really on screen
  4. tus upload to tus_endpoint with Upload-Metadata session_id + token; the stream then reports
     upload_progress, uploaded, analysing and finally `done` with live / spoof / retry.
tusd calls /hooks/tus; a separate `liveness-worker` process runs the analysis.

Run: uv run uvicorn liveness_service.api.http.app:app --host 0.0.0.0 --port 8000
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from liveness_service.api.http.routes import events, health, sessions, tus_hooks, uploads
from liveness_service.core import tusd_client
from liveness_service.core.logger import LoggerProvider
from liveness_service.core.redis import RedisManager
from liveness_service.core.settings import get_settings
from liveness_service.exceptions import RateLimited, SessionError
from liveness_service.repositories.session_repository import SessionRepository
from liveness_service.services.sessions import SessionService
from liveness_service.services.uploads import UploadService


@asynccontextmanager
async def lifespan(app):
    settings = get_settings()  # fail at startup, not on the first request, if settings or Vault are wrong
    repository = SessionRepository(settings.session_ttl_seconds)
    redis = RedisManager(LoggerProvider(), settings).add(repository)
    await redis.connect()
    app.state.sessions = SessionService(repository, settings)
    app.state.uploads = UploadService(repository, app.state.sessions)
    app.state.tusd = tusd_client.create_client()
    yield
    await app.state.tusd.aclose()
    await redis.disconnect()


app = FastAPI(title="Liveness Service", lifespan=lifespan)
for module in (health, sessions, events, uploads, tus_hooks):
    app.include_router(module.router)


@app.exception_handler(SessionError)
async def session_error(request: Request, err: SessionError):
    headers = {"Retry-After": str(err.retry_after_seconds)} if isinstance(err, RateLimited) else None
    return JSONResponse({"detail": err.detail}, status_code=err.status_code, headers=headers)
