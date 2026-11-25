"""Liveness session API on fake Redis: lifecycle, tokens, just-in-time reveal, display log, tusd hooks,
single use, rate limiting and the analysis job."""
import json
import threading
import time

import httpx
import pytest

from liveness_service.core.security import issue_session_token
from liveness_service.enums import CheckOutcome
from liveness_service.repositories.audit_repository import SQLiteAuditStore
from liveness_service.services import analysis as analysis_module
from liveness_service.services.analysis import AnalysisService

SESSIONS = "/v1/liveness/sessions"


def create(client, **body):
    response = client.post(SESSIONS, json=body or None)
    assert response.status_code == 201, response.text
    return response.json()


def auth(token):
    return {"Authorization": f"Bearer {token}"}


def read_events(client, session, until="done", stop_after=None):
    """Read SSE events (name, data, received_at) until event `until` or `stop_after` events."""
    events = []
    with client.stream("GET", f"{SESSIONS}/{session['session_id']}/events", headers=auth(session["token"])) as r:
        assert r.status_code == 200, r.read()
        name = None
        for line in r.iter_lines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                events.append((name, json.loads(line[6:]), time.monotonic()))
                if name == until or (stop_after and len(events) >= stop_after):
                    break
    return events


def hook(client, kind, session=None, upload_id=None, **upload):
    metadata = {"session_id": session["session_id"], "token": session["token"]} if session else {}
    body = {"Type": kind, "Event": {"Upload": {"ID": upload_id, "MetaData": metadata, **upload}}}
    response = client.post("/hooks/tus", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def stored(redis_inspect, session):
    raw = redis_inspect.hgetall(f"liveness:session:{session['session_id']}")
    return {k: json.loads(v) if k in ("colour_sequence", "display_log", "result") else v for k, v in raw.items()}


def reveal_all(client, session):
    """Open the stream until every colour was revealed (3 with the test settings)."""
    return read_events(client, session, until=None, stop_after=4)  # capture + 3 colours


# --- create / status / tokens ------------------------------------------------------------------------------

def test_create_returns_token_but_never_the_sequence(client, redis_inspect):
    response = client.post(SESSIONS, json={"device_id": "phone-1"})
    body = response.json()
    assert response.status_code == 201 and set(body) == {"session_id", "token", "tus_endpoint", "expires_at"}
    assert body["tus_endpoint"] == "http://testserver/v1/liveness/uploads"
    sequence = stored(redis_inspect, body)["colour_sequence"]
    assert len(sequence) == 3 and not any(step["colour"] in response.text for step in sequence)
    assert stored(redis_inspect, body)["status"] == "created"
    assert 0 < redis_inspect.ttl(f"liveness:session:{body['session_id']}") <= 600


def test_status_needs_the_sessions_own_token(client):
    a, b = create(client), create(client)
    url = f"{SESSIONS}/{a['session_id']}"
    assert client.get(url).status_code == 401
    assert client.get(url, headers=auth("123.forged")).status_code == 401
    assert client.get(url, headers=auth(b["token"])).status_code == 401  # another session's token
    assert client.get(url, headers=auth(a["token"])).json() == {"status": "created"}


def test_expired_session_is_gone(client, redis_inspect):
    session = create(client)
    expired = issue_session_token(session["session_id"], int(time.time()) - 1)
    assert client.get(f"{SESSIONS}/{session['session_id']}", headers=auth(expired)).status_code == 410
    redis_inspect.delete(f"liveness:session:{session['session_id']}")  # TTL ran out
    assert client.get(f"{SESSIONS}/{session['session_id']}", headers=auth(session["token"])).status_code == 410


def test_rate_limit_per_device(make_client):
    client = make_client(RATE_LIMIT_MAX_SESSIONS=2)
    for _ in range(2):
        create(client, device_id="phone-1")
    response = client.post(SESSIONS, json={"device_id": "phone-1"})
    assert response.status_code == 429 and int(response.headers["Retry-After"]) > 0


def test_attempt_counts_sessions_of_the_device(client, redis_inspect):
    first, second = create(client, device_id="phone-9"), create(client, device_id="phone-9")
    assert stored(redis_inspect, first)["attempt"] == "1" and stored(redis_inspect, second)["attempt"] == "2"


# --- SSE: just-in-time colour reveal ------------------------------------------------------------------------

def test_colours_are_revealed_one_by_one_on_schedule(client, redis_inspect):
    session = create(client)
    events = reveal_all(client, session)
    assert [name for name, _, _ in events] == ["capture", "colour", "colour", "colour"]
    assert events[0][1] == {"lead_in_ms": 0, "steps": 3}
    sequence = stored(redis_inspect, session)["colour_sequence"]
    for (_, data, _), step in zip(events[1:], sequence):
        assert data == {k: step[k] for k in ("index", "colour", "rgb", "duration_ms")}
    # each colour arrives only when it's due: no earlier than the previous one's duration
    for (_, _, t0), (_, _, t1), step in zip(events[1:], events[2:], sequence):
        assert t1 - t0 >= step["duration_ms"] / 1000 - 0.05
    assert stored(redis_inspect, session)["revealed_index"] == "2"
    assert stored(redis_inspect, session)["status"] == "capturing"


def test_reconnecting_continues_without_repeating_colours(client):
    session = create(client)
    first = read_events(client, session, until=None, stop_after=2)   # capture + colour 0, then drop
    second = read_events(client, session, until=None, stop_after=2)  # capture + the next colour
    assert first[1][1]["index"] == 0 and second[1][1]["index"] == 1


# --- display log --------------------------------------------------------------------------------------------

def display_log(indices, start=1_000):
    return {"recording_started_at_ms": start, "steps": [{"index": i, "shown_at_ms": start + 500 + 400 * i}
                                                        for i in indices]}


def test_display_log_rules(client):
    session = create(client)
    url = f"{SESSIONS}/{session['session_id']}/display-log"
    assert client.post(url, json=display_log([0])).status_code == 401
    assert client.post(url, json=display_log([0]), headers=auth(session["token"])).status_code == 409  # no capture
    read_events(client, session, until=None, stop_after=2)  # only colour 0 revealed
    assert client.post(url, json=display_log([0, 1]), headers=auth(session["token"])).status_code == 409
    assert client.post(url, json=display_log([0]), headers=auth(session["token"])).status_code == 204
    assert client.post(url, json=display_log([0]), headers=auth(session["token"])).status_code == 409  # once


# --- tusd hooks, single use, analysis, done -----------------------------------------------------------------

def test_pre_create_validates_session_and_is_single_use(client):
    session = create(client)
    assert hook(client, "pre-create")["HTTPResponse"]["StatusCode"] == 401  # no metadata
    forged = {**session, "token": issue_session_token(session["session_id"], int(time.time()) + 60)[:-2] + "xx"}
    assert hook(client, "pre-create", forged)["RejectUpload"] is True
    assert hook(client, "pre-create", session) == {}
    second = hook(client, "pre-create", session)
    assert second["RejectUpload"] is True and second["HTTPResponse"]["StatusCode"] == 409


def test_full_flow_hooks_to_done(client, redis_inspect, monkeypatch, tmp_path):
    session = create(client)
    reveal_all(client, session)
    url = f"{SESSIONS}/{session['session_id']}"
    assert client.post(f"{url}/display-log", json=display_log([0, 1, 2]), headers=auth(session["token"])).status_code == 204

    events, listener = [], threading.Thread(target=lambda: events.extend(read_events(client, session)))
    listener.start()
    time.sleep(0.3)  # let the stream subscribe before the hooks publish
    assert hook(client, "pre-create", session) == {}
    hook(client, "post-receive", session, "up1", Offset=50, Size=100)
    hook(client, "post-finish", session, "up1", Offset=100, Size=100)
    hook(client, "post-receive", session, "up1", Offset=100, Size=100)  # late, as real tusd does
    assert stored(redis_inspect, session)["status"] == "uploaded"  # didn't move back to uploading
    job = json.loads(redis_inspect.rpop("liveness:jobs"))
    assert job == {"session_id": session["session_id"], "upload_id": "up1"}

    # run the worker's job with the verifier stubbed (the real one is exercised in test_colour_signals)
    monkeypatch.setattr(analysis_module, "verify_video",
                        lambda path, shown, config, regions: (CheckOutcome.PASS, {"correlation": 0.9}))
    audit = SQLiteAuditStore(tmp_path / "audit.sqlite3")
    service = AnalysisService(client.app.state.sessions.repo, client.app.state.sessions, audit,
                              client.app.state.sessions.settings, regions=None)
    client.portal.call(service.run, job)
    listener.join(timeout=10)

    # all colours were already revealed, so no `capture`; the late post-receive published nothing
    assert [name for name, _, _ in events] == ["upload_progress", "uploaded", "analysing", "done"]
    assert events[0][1] == {"offset": 50, "size": 100} and events[-1][1] == {"decision": "live"}
    assert client.get(url, headers=auth(session["token"])).json() == {"status": "done", "decision": "live"}
    assert client.get(f"{url}/events", headers=auth(session["token"])).status_code == 410  # finished
    record = audit.get(session["session_id"])
    assert record["decision"] == "live" and record["scores"] == {"correlation": 0.9}


def test_analysis_without_display_log_asks_for_retry(client, tmp_path):
    session = create(client)
    hook(client, "pre-create", session)
    hook(client, "post-finish", session, "up2", Offset=10, Size=10)
    audit = SQLiteAuditStore(tmp_path / "audit.sqlite3")
    service = AnalysisService(client.app.state.sessions.repo, client.app.state.sessions, audit,
                              client.app.state.sessions.settings, regions=None)
    client.portal.call(service.run, {"session_id": session["session_id"], "upload_id": "up2"})
    assert audit.get(session["session_id"])["error"] == "missing display log"
    status = client.get(f"{SESSIONS}/{session['session_id']}", headers=auth(session["token"])).json()
    assert status == {"status": "done", "decision": "retry"}


def test_terminated_upload_ends_with_retry(client):
    session = create(client)
    hook(client, "pre-create", session)
    hook(client, "post-terminate", session, "up3")
    status = client.get(f"{SESSIONS}/{session['session_id']}", headers=auth(session["token"])).json()
    assert status == {"status": "failed", "decision": "retry"}


def test_hooks_for_unknown_or_malformed_ids_are_ignored(client, redis_inspect):
    hook(client, "post-finish", {"session_id": "../../etc", "token": "x"}, "up")
    hook(client, "post-finish", {"session_id": "6f1c0f1e-0000-4000-8000-000000000000", "token": "x"}, "../x")
    assert redis_inspect.llen("liveness:jobs") == 0


# --- tus proxy ----------------------------------------------------------------------------------------------

@pytest.fixture
def fake_tusd(client):
    seen, real = [], client.app.state.tusd

    def handler(request):
        seen.append(request)
        if request.method == "POST":
            return httpx.Response(201, headers={"Location": "http://tusd:8080/files/abc123", "Tus-Resumable": "1.0.0"})
        return httpx.Response(204, headers={"Upload-Offset": str(len(request.content)), "Tus-Resumable": "1.0.0"})

    client.app.state.tusd = httpx.AsyncClient(base_url="http://tusd:8080/files/", transport=httpx.MockTransport(handler))
    yield seen
    client.app.state.tusd = real


def test_tus_upload_is_proxied_to_tusd(client, fake_tusd):
    created = client.post("/v1/liveness/uploads", headers={"Tus-Resumable": "1.0.0", "Upload-Length": "5"})
    assert created.status_code == 201
    assert created.headers["location"] == "http://testserver/v1/liveness/uploads/abc123"
    patched = client.patch("/v1/liveness/uploads/abc123", content=b"hello",
                           headers={"Tus-Resumable": "1.0.0", "Upload-Offset": "0",
                                    "Content-Type": "application/offset+octet-stream"})
    assert patched.status_code == 204 and fake_tusd[1].content == b"hello"


def test_health_and_openapi(client):
    assert client.get("/health").json() == {"status": "ok"}
    schemas = client.get("/openapi.json").json()["components"]["schemas"]
    assert {"CreateSessionResponse", "SessionStatusResponse", "DisplayLog", "TusdHook"} <= schemas.keys()
