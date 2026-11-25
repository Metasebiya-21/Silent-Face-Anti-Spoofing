"""HTTP API for the mobile app: video liveness under /v1/liveness (see video.py).

Run: TUSD_URL=http://localhost:8021/files/ TUSD_DATA_DIR=./data \
     uv run uvicorn silent_face_anti_spoofing.api:app --host 0.0.0.0 --port 8000
Set LIVENESS_API_KEY to require the app to send it in the X-API-Key header.
"""
import os
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from silent_face_anti_spoofing.api import video
from silent_face_anti_spoofing.checker import LivenessChecker


@asynccontextmanager
async def lifespan(app):
    app.state.checker = LivenessChecker()  # load models once per worker, not per request
    app.state.tasks = set()                # running video checks
    app.state.tusd = httpx.AsyncClient(base_url=os.environ.get("TUSD_URL", "http://localhost:8080/files/"),
                                       timeout=httpx.Timeout(30, connect=5))
    yield
    await app.state.tusd.aclose()


app = FastAPI(title="Silent Face Anti-Spoofing", lifespan=lifespan)
app.include_router(video.router)


@app.get("/health")
def health():
    return {"status": "ok"}
