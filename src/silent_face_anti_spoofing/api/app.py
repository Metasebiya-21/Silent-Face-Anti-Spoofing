"""HTTP API for peripheral devices.

- POST /v1/liveness: one image in, verdict out.
- Video: uploaded through tusd, verdict streamed over SSE (see video.py).

Run: uv run uvicorn silent_face_anti_spoofing.api:app --host 0.0.0.0 --port 8000
Set LIVENESS_API_KEY to require devices to send it in the X-API-Key header.
"""
from contextlib import asynccontextmanager
from dataclasses import asdict

import cv2
import numpy as np
from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile

from silent_face_anti_spoofing.api import video
from silent_face_anti_spoofing.api.auth import authorized
from silent_face_anti_spoofing.checker import LivenessChecker

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_PIXELS = 4096 * 4096


@asynccontextmanager
async def lifespan(app):
    app.state.checker = LivenessChecker()  # load models once per worker, not per request
    app.state.tasks = set()                # running video checks
    yield


app = FastAPI(title="Silent Face Anti-Spoofing", lifespan=lifespan)
app.include_router(video.router)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/liveness")
def liveness(request: Request, image: UploadFile = File(...), x_api_key: str | None = Header(None)):
    if not authorized(x_api_key):
        raise HTTPException(401, "invalid or missing X-API-Key")

    data = image.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"image larger than {MAX_UPLOAD_BYTES} bytes")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "could not decode image")
    if img.shape[0] * img.shape[1] > MAX_PIXELS:
        raise HTTPException(413, f"image has more than {MAX_PIXELS} pixels")

    result = request.app.state.checker.check(img)
    if result is None:
        return {"face_found": False}
    return {"face_found": True, **asdict(result)}
