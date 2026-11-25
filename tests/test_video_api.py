import json
import time
from contextlib import contextmanager
from pathlib import Path

import cv2
import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from liveness_service.api.http.app import app
from liveness_service.services import event_stream

SAMPLES = Path(__file__).resolve().parents[1] / "images" / "sample"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def data_dir(tmp_path, setenv):
    setenv("TUSD_DATA_DIR", str(tmp_path))
    return tmp_path


def write_video(path, frame, count=12):
    """Write `count` copies of frame as a video at path (no extension, like tusd's filestore)."""
    avi = path.with_suffix(".avi")
    writer = cv2.VideoWriter(str(avi), cv2.VideoWriter_fourcc(*"MJPG"), 10, (frame.shape[1], frame.shape[0]))
    for _ in range(count):
        writer.write(frame)
    writer.release()
    avi.rename(path)


def hook(client, kind, upload_id=None, headers=None, **upload):
    body = {"Type": kind, "Event": {"Upload": {"ID": upload_id, **upload},
                                    "HTTPRequest": {"Header": headers or {}}}}
    response = client.post("/tusd/hooks", json=body)
    assert response.status_code == 200
    return response.json()


def read_events(client, upload_id, **headers):
    deadline = time.time() + 60  # don't hang the test run if a final event never arrives
    while not (upload_id in event_stream.sessions and event_stream.sessions[upload_id].done):
        assert time.time() < deadline, "no final event"
        time.sleep(0.05)
    text = client.get(f"/v1/liveness/{upload_id}/events", headers=headers).text
    events = []
    for block in text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines() if not line.startswith(":"))
        events.append((int(fields["id"]), fields["event"], json.loads(fields["data"])))
    return events


@pytest.mark.parametrize("upload_id, image, is_real", [("realvid", "image_T1.jpg", True),
                                                       ("fakevid", "image_F1.jpg", False)])
def test_upload_flow_streams_progress_then_result(client, data_dir, upload_id, image, is_real):
    write_video(data_dir / upload_id, cv2.imread(str(SAMPLES / image)))
    hook(client, "post-receive", upload_id, Offset=100, Size=200)
    hook(client, "post-finish", upload_id, Offset=200, Size=200)

    events = read_events(client, upload_id)
    assert [name for _, name, _ in events] == ["progress", "processing", "result"]
    assert events[0][2] == {"offset": 100, "size": 200}
    result = events[-1][2]
    assert result["face_found"] is True
    assert result["is_real"] is is_real
    assert result["frames_with_face"] == result["frames_sampled"] == 10


def test_reconnect_with_last_event_id_replays_only_missed_events(client, data_dir):
    write_video(data_dir / "resume", cv2.imread(str(SAMPLES / "image_T1.jpg")))
    hook(client, "post-finish", "resume")
    assert [name for _, name, _ in read_events(client, "resume")] == ["processing", "result"]
    assert [name for _, name, _ in read_events(client, "resume", **{"Last-Event-ID": "0"})] == ["result"]


def test_no_face_and_unreadable_video(client, data_dir):
    write_video(data_dir / "blank", np.full((640, 480, 3), 128, np.uint8))
    hook(client, "post-finish", "blank")
    assert read_events(client, "blank")[-1][1:] == ("result", {"face_found": False})

    (data_dir / "junk").write_bytes(b"not a video")
    hook(client, "post-finish", "junk")
    assert read_events(client, "junk")[-1][1:] == ("error", {"detail": "could not open video"})


def test_progress_arriving_after_the_result_is_dropped(client, data_dir):
    # seen with real tusd: post-receive hooks landed after post-finish
    write_video(data_dir / "late", cv2.imread(str(SAMPLES / "image_T1.jpg")))
    hook(client, "post-finish", "late")
    read_events(client, "late")
    hook(client, "post-receive", "late", Offset=200, Size=200)
    assert [name for _, name, _ in read_events(client, "late")] == ["processing", "result"]


def test_terminated_upload_ends_stream(client):
    hook(client, "post-terminate", "gone")
    assert read_events(client, "gone")[-1][1:] == ("error", {"detail": "upload terminated"})


def test_forged_hook_with_path_in_id_is_ignored(client):
    hook(client, "post-finish", "../../etc/passwd")
    assert "../../etc/passwd" not in event_stream.sessions
    assert client.get("/v1/liveness/..%2Fx/events").status_code in (400, 404)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


@contextmanager
def tusd_handler(handler):
    """Point the app at an in-memory tusd for the duration of the block."""
    real = app.state.tusd
    app.state.tusd = httpx.AsyncClient(base_url="http://tusd:8080/files/", transport=httpx.MockTransport(handler))
    try:
        yield
    finally:
        app.state.tusd = real


@pytest.fixture
def fake_tusd(client):
    """Replace the app's tusd client with an in-memory tusd; records what reached it."""
    seen = []

    def handler(request):
        seen.append(request)
        if request.method == "POST":
            return httpx.Response(201, headers={"Location": "http://tusd:8080/files/abc123", "Tus-Resumable": "1.0.0"})
        if request.method == "PATCH":
            return httpx.Response(204, headers={"Upload-Offset": str(len(request.content)), "Tus-Resumable": "1.0.0"})
        return httpx.Response(200, headers={"Upload-Offset": "5", "Upload-Length": "10", "Tus-Resumable": "1.0.0"})

    with tusd_handler(handler):
        yield seen


def test_tus_is_proxied_under_v1_liveness(client, fake_tusd):
    tus = {"Tus-Resumable": "1.0.0"}
    created = client.post("/v1/liveness", headers={**tus, "Upload-Length": "10"})
    assert created.status_code == 201
    assert created.headers["location"] == "http://testserver/v1/liveness/abc123"  # tusd's URL never leaks

    patched = client.patch("/v1/liveness/abc123", content=b"hello",
                           headers={**tus, "Upload-Offset": "0", "Content-Type": "application/offset+octet-stream"})
    assert patched.status_code == 204
    assert patched.headers["upload-offset"] == "5"

    resumed = client.head("/v1/liveness/abc123", headers=tus)
    assert resumed.headers["upload-offset"] == "5"

    create, patch, head = fake_tusd
    assert (create.method, str(create.url), create.headers["upload-length"]) == ("POST", "http://tusd:8080/files/", "10")
    assert (patch.method, str(patch.url), patch.content) == ("PATCH", "http://tusd:8080/files/abc123", b"hello")
    assert patch.headers["upload-offset"] == "0"
    assert head.method == "HEAD"


def test_tus_proxy_rejects_bad_requests(client, fake_tusd, setenv):
    assert client.patch("/v1/liveness/bad.id", content=b"x").status_code == 400
    setenv("LIVENESS_API_KEY", "secret")
    assert client.post("/v1/liveness", headers={"Upload-Length": "10"}).status_code == 401
    assert fake_tusd == []  # nothing reached tusd


def test_tusd_down_gives_502(client):
    def down(request):
        raise httpx.ConnectError("refused")
    with tusd_handler(down):
        assert client.post("/v1/liveness", headers={"Upload-Length": "10"}).status_code == 502


def test_api_key_on_tus_create_and_sse(client, setenv):
    setenv("LIVENESS_API_KEY", "secret")
    assert hook(client, "pre-create")["RejectUpload"] is True
    assert hook(client, "pre-create", headers={"X-Api-Key": ["wrong"]})["RejectUpload"] is True
    assert hook(client, "pre-create", headers={"X-Api-Key": ["secret"]}) == {}
    assert client.get("/v1/liveness/abc/events").status_code == 401


def test_invalid_last_event_id_is_rejected(client):
    assert client.get("/v1/liveness/abc/events", headers={"Last-Event-ID": "-1"}).status_code == 422


def test_openapi_documents_models(client):
    schemas = client.get("/openapi.json").json()["components"]["schemas"]
    assert {"HealthResponse", "ErrorResponse", "TusdHook", "TusdHookResponse"} <= schemas.keys()
