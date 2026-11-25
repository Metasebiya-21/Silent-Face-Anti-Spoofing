from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from silent_face_anti_spoofing.api import app
from silent_face_anti_spoofing.api.app import MAX_UPLOAD_BYTES

SAMPLES = Path(__file__).resolve().parents[1] / "images" / "sample"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # `with` runs the lifespan, which loads the models
        yield c


def post(client, data, **headers):
    return client.post("/v1/liveness", files={"image": ("frame.jpg", data, "image/jpeg")}, headers=headers)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


@pytest.mark.parametrize("name, is_real", [("image_T1.jpg", True), ("image_F1.jpg", False)])
def test_sample_verdicts(client, name, is_real):
    body = post(client, (SAMPLES / name).read_bytes()).json()
    assert body["face_found"] is True
    assert body["is_real"] is is_real
    assert len(body["bbox"]) == 4


def test_no_face(client):
    blank = cv2.imencode(".jpg", np.full((640, 480, 3), 128, np.uint8))[1].tobytes()
    assert post(client, blank).json() == {"face_found": False}


def test_rejects_bad_input(client):
    assert post(client, b"not an image").status_code == 400
    assert post(client, b"0" * (MAX_UPLOAD_BYTES + 1)).status_code == 413


def test_api_key(client, monkeypatch):
    monkeypatch.setenv("LIVENESS_API_KEY", "secret")
    image = (SAMPLES / "image_T1.jpg").read_bytes()
    assert post(client, image).status_code == 401
    assert post(client, image, **{"X-API-Key": "wrong"}).status_code == 401
    assert post(client, image, **{"X-API-Key": "secret"}).status_code == 200
