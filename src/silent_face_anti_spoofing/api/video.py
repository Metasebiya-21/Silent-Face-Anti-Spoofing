"""Video liveness: the app only talks to /v1/liveness; uploads are proxied to tusd, results come back over SSE.

Flow for the app (every request carries X-API-Key):
  1. tus create: POST /v1/liveness. The Location header is /v1/liveness/{upload_id}.
  2. GET /v1/liveness/{upload_id}/events (SSE) and keep it open.
  3. tus PATCH /v1/liveness/{upload_id} chunks; HEAD it to get the offset to resume from.
     The stream sends `progress`, then `processing`, then one final `result` or `error`, and closes.
     Reconnect with Last-Event-ID to resume the stream.

tusd (TUSD_URL) calls POST /tusd/hooks at each stage; after post-finish the video is checked.
Keep tusd and /tusd/hooks reachable only internally.
"""
import asyncio
import json
import logging
import os
import re
from dataclasses import asdict
from pathlib import Path

import httpx
from fastapi import APIRouter, Header, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from silent_face_anti_spoofing.api.auth import authorized

log = logging.getLogger(__name__)
router = APIRouter()

UPLOAD_ID = re.compile(r"[A-Za-z0-9_-]{1,128}")  # tusd filestore IDs are hex; this also blocks path tricks
SESSION_TTL_SECONDS = 3600
KEEPALIVE_SECONDS = 15
# headers that describe one hop, not the tus request; Content-Length is kept on requests (tusd needs it on PATCH)
HOP_BY_HOP = {"host", "connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade",
              "proxy-authorization", "proxy-authenticate"}


async def proxy_to_tusd(request, path):
    if not authorized(request.headers.get("x-api-key")):
        raise HTTPException(401, "invalid or missing X-API-Key")
    headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP}
    body = request.stream() if request.method in ("POST", "PATCH") else None  # stream chunks, don't buffer them
    try:
        r = await request.app.state.tusd.request(request.method, path, headers=headers, content=body)
    except httpx.HTTPError:
        log.exception("tusd request failed")
        raise HTTPException(502, "upload server unavailable")
    out = {k: v for k, v in r.headers.items() if k.lower() not in HOP_BY_HOP | {"content-length"}}
    if "location" in out:  # point the app at our URL, not tusd's
        upload_id = out["location"].rstrip("/").rsplit("/", 1)[-1]
        out["location"] = str(request.url_for("tus_upload", upload_id=upload_id))
    return Response(r.content, r.status_code, headers=out)


@router.api_route("/v1/liveness", methods=["POST", "OPTIONS"])
async def tus_create(request: Request):
    return await proxy_to_tusd(request, "")


@router.api_route("/v1/liveness/{upload_id}", methods=["HEAD", "PATCH", "DELETE"], name="tus_upload")
async def tus_upload(upload_id: str, request: Request):
    if not UPLOAD_ID.fullmatch(upload_id):
        raise HTTPException(400, "invalid upload id")
    return await proxy_to_tusd(request, upload_id)


class UploadEvents:
    """Every event published for one upload, kept so late or reconnecting SSE clients can replay them."""

    def __init__(self):
        self.events = []  # (event name, data dict); an event's SSE id is its index
        self.done = False
        self.changed = asyncio.Event()

    def publish(self, event, data, final=False):
        if self.done:  # tusd sends post-receive asynchronously; it can arrive after post-finish's result
            return
        self.events.append((event, data))
        self.done = self.done or final
        self.changed.set()
        self.changed = asyncio.Event()  # current waiters woke on the old one; later waiters get a fresh one


# shortcut: in-memory, so run a single uvicorn worker; move to Redis pub/sub to scale out or survive restarts
sessions = {}


def session(upload_id):
    if upload_id not in sessions:
        sessions[upload_id] = UploadEvents()
        # shortcut: sessions expire SESSION_TTL_SECONDS after creation, even if the upload is still running
        asyncio.get_running_loop().call_later(SESSION_TTL_SECONDS, sessions.pop, upload_id, None)
    return sessions[upload_id]


def upload_path(upload_id):
    # tusd filestore writes upload <id> to <dir>/<id>; built from our config, not the hook's Storage.Path,
    # so a forged hook can't point us at an arbitrary file
    return Path(os.environ.get("TUSD_DATA_DIR", "./data")) / upload_id


@router.post("/tusd/hooks")
async def tusd_hook(hook: dict, request: Request):
    kind = hook.get("Type")
    event = hook.get("Event", {})
    upload = event.get("Upload", {})

    if kind == "pre-create":  # the only hook that can reject an upload
        headers = event.get("HTTPRequest", {}).get("Header", {})
        if not authorized((headers.get("X-Api-Key") or [None])[0]):  # tusd sends header values as lists
            return {"RejectUpload": True, "HTTPResponse": {"StatusCode": 401, "Body": "invalid or missing X-API-Key"}}
        return {}

    upload_id = upload.get("ID") or ""
    if not UPLOAD_ID.fullmatch(upload_id):
        return {}
    events = session(upload_id)
    if kind == "post-receive":
        events.publish("progress", {"offset": upload.get("Offset"), "size": upload.get("Size")})
    elif kind == "post-finish":
        events.publish("processing", {})
        request.app.state.tasks.add(task := asyncio.create_task(check_upload(request.app, upload_id, events)))
        task.add_done_callback(request.app.state.tasks.discard)  # keep a reference so the task isn't GC'd
    elif kind == "post-terminate":
        events.publish("error", {"detail": "upload terminated"}, final=True)
    return {}


async def check_upload(app, upload_id, events):
    try:
        result = await asyncio.to_thread(app.state.checker.check_video, upload_path(upload_id))
    except ValueError as err:  # not a readable video
        events.publish("error", {"detail": str(err)}, final=True)
        return
    except Exception:
        log.exception("liveness check failed for upload %s", upload_id)
        events.publish("error", {"detail": "liveness check failed"}, final=True)
        return
    data = {"face_found": False} if result is None else {"face_found": True, **asdict(result)}
    events.publish("result", data, final=True)


@router.get("/v1/liveness/{upload_id}/events")
async def liveness_events(upload_id: str, x_api_key: str | None = Header(None),
                          last_event_id: str | None = Header(None)):
    if not authorized(x_api_key):
        raise HTTPException(401, "invalid or missing X-API-Key")
    if not UPLOAD_ID.fullmatch(upload_id):
        raise HTTPException(400, "invalid upload id")
    events = session(upload_id)  # may open before tusd's first hook arrives for this upload
    start = int(last_event_id) + 1 if last_event_id and last_event_id.isdigit() else 0

    async def stream():
        i = start
        while True:
            while i < len(events.events):
                name, data = events.events[i]
                yield f"id: {i}\nevent: {name}\ndata: {json.dumps(data)}\n\n"
                i += 1
            if events.done:
                return
            try:
                await asyncio.wait_for(events.changed.wait(), KEEPALIVE_SECONDS)
            except TimeoutError:
                yield ": keep-alive\n\n"  # stops proxies from closing an idle connection

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
