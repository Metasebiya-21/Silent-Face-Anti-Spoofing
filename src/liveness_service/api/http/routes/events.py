"""SSE stream of one upload's events: progress, processing, then a final result or error."""
import asyncio
import json

from fastapi import APIRouter, Header
from fastapi.responses import StreamingResponse

from liveness_service.core import config
from liveness_service.core.security import require_api_key, require_upload_id
from liveness_service.services import event_stream

router = APIRouter(prefix="/v1/liveness", tags=["events"])


@router.get("/{upload_id}/events")
async def liveness_events(upload_id: str, x_api_key: str | None = Header(None),
                          last_event_id: str | None = Header(None)):
    require_api_key(x_api_key)
    require_upload_id(upload_id)
    events = event_stream.get(upload_id)
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
                await asyncio.wait_for(events.changed.wait(), config.KEEPALIVE_SECONDS)
            except TimeoutError:
                yield ": keep-alive\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
