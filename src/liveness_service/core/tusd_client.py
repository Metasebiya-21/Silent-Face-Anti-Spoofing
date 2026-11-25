"""Forwards the app's tus requests to tusd, so the app never talks to tusd directly."""
import logging

import httpx
from fastapi import HTTPException, Response

from liveness_service.core.settings import get_settings

log = logging.getLogger(__name__)

# headers that describe one hop, not the tus request; Content-Length is kept on requests (tusd needs it on PATCH)
HOP_BY_HOP = {"host", "connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade",
              "proxy-authorization", "proxy-authenticate"}


def create_client():
    return httpx.AsyncClient(base_url=str(get_settings().tusd_url), timeout=httpx.Timeout(30, connect=5))


async def forward(request, path, upload_url_for):
    """Send `request` to tusd at `path`; upload_url_for(upload_id) builds the URL the app should use instead."""
    headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP}
    body = request.stream() if request.method in ("POST", "PATCH") else None  # stream chunks, don't buffer them
    try:
        r = await request.app.state.tusd.request(request.method, path, headers=headers, content=body)
    except httpx.HTTPError:
        log.exception("tusd request failed")
        raise HTTPException(502, "upload server unavailable")
    out = {k: v for k, v in r.headers.items() if k.lower() not in HOP_BY_HOP | {"content-length"}}
    if "location" in out:  # point the app at our URL, not tusd's
        out["location"] = upload_url_for(out["location"].rstrip("/").rsplit("/", 1)[-1])
    return Response(r.content, r.status_code, headers=out)
