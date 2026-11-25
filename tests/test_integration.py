"""End to end against the docker compose stack: create session -> SSE reveal -> tus upload -> hooks ->
worker -> SSE `done`.

    docker compose up --build -d
    RUN_INTEGRATION=1 uv run pytest tests/test_integration.py -v

The capture is synthetic: the sample photo, with its face tinted by each revealed colour 80 ms after it was
"shown". Until the verifier TODOs are implemented the worker can't score it and the decision is `retry`;
this test only checks that the whole pipeline reaches `done`.
"""
import base64
import json
import os
import threading
import time
from pathlib import Path

import cv2
import httpx
import numpy as np
import pytest

pytestmark = [pytest.mark.integration,
              pytest.mark.skipif(os.environ.get("RUN_INTEGRATION") != "1", reason="set RUN_INTEGRATION=1")]

BASE_URL = os.environ.get("LIVENESS_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("LIVENESS_API_KEY", "change-me")
SAMPLE = Path(__file__).resolve().parents[1] / "images" / "sample" / "image_T1.jpg"
FACE_BOX = (119, 113, 170, 240)  # x, y, w, h of the face in the sample photo (YuNet)


def now_ms():
    return time.time() * 1000


def listen(session, events):
    headers = {"Authorization": f"Bearer {session['token']}"}
    with httpx.stream("GET", f"{BASE_URL}/v1/liveness/sessions/{session['session_id']}/events",
                      headers=headers, timeout=120) as response:
        name = None
        for line in response.iter_lines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                events.append((name, json.loads(line[6:]), now_ms()))
                if name == "done":
                    return


def synthetic_capture(path, recording_started_ms, shown, lag_ms=80, fps=30):
    """Video of the sample face, lit by each shown colour from shown_at + lag."""
    image = cv2.imread(str(SAMPLE)).astype(float)
    x, y, w, h = FACE_BOX
    end = shown[-1][0] + 800
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (image.shape[1], image.shape[0]))
    for t in np.arange(recording_started_ms, end, 1000 / fps):
        frame = image.copy()
        current = [rgb for at, rgb in shown if at + lag_ms <= t]
        if current:
            frame[y:y + h, x:x + w] += 0.15 * np.array(current[-1][::-1])  # BGR
        writer.write(np.clip(frame, 0, 255).astype(np.uint8))
    writer.release()


def metadata(**values):
    return ",".join(f"{k} {base64.b64encode(v.encode()).decode()}" for k, v in values.items())


def test_session_to_done(tmp_path):
    created = httpx.post(f"{BASE_URL}/v1/liveness/sessions", headers={"X-API-Key": API_KEY},
                         json={"device_id": "integration-test"})
    assert created.status_code == 201, created.text
    session = created.json()

    events = []
    listener = threading.Thread(target=listen, args=(session, events), daemon=True)
    listener.start()
    deadline = time.time() + 30
    while sum(name == "colour" for name, _, _ in events) < 10 and time.time() < deadline:
        time.sleep(0.05)
    capture = next(at for name, _, at in events if name == "capture")
    colours = [(at, data) for name, data, at in events if name == "colour"]
    assert len(colours) == 10, events

    video = tmp_path / "capture.avi"
    synthetic_capture(video, capture, [(at, tuple(data["rgb"])) for at, data in colours])
    auth = {"Authorization": f"Bearer {session['token']}"}
    log = {"recording_started_at_ms": int(capture),
           "steps": [{"index": data["index"], "shown_at_ms": int(at)} for at, data in colours]}
    assert httpx.post(f"{BASE_URL}/v1/liveness/sessions/{session['session_id']}/display-log",
                      json=log, headers=auth).status_code == 204

    tus = {"Tus-Resumable": "1.0.0", "X-API-Key": API_KEY}
    body = video.read_bytes()
    upload = httpx.post(session["tus_endpoint"], headers={
        **tus, "Upload-Length": str(len(body)),
        "Upload-Metadata": metadata(session_id=session["session_id"], token=session["token"])})
    assert upload.status_code == 201, upload.text
    patched = httpx.patch(upload.headers["location"], content=body, headers={
        **tus, "Upload-Offset": "0", "Content-Type": "application/offset+octet-stream"})
    assert patched.status_code == 204, patched.text

    listener.join(timeout=90)
    names = [name for name, _, _ in events]
    assert names[-1] == "done", names
    assert {"uploaded", "analysing"} <= set(names)
    assert events[-1][1]["decision"] in {"live", "spoof", "retry"}
