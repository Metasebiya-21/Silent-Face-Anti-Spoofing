"""SSE stream of one upload's events: progress, processing, then a final result or error."""
import asyncio
import json
from typing import Annotated

from fastapi import APIRouter, Header
from fastapi.responses import StreamingResponse

from liveness_service.core.security import require_api_key, require_upload_id
from liveness_service.core.settings import get_settings
from liveness_service.schemas.inbound.events import EventStreamHeaders
from liveness_service.schemas.outbound.http import error_responses
from liveness_service.services import event_stream

router = APIRouter(prefix="/v1/liveness", tags=["events"])

SSE_DOC = {200: {"content": {"text/event-stream": {}},
                 "description": "`progress` (Progress), `processing`, then one final `result` (Result) or "
                                "`error` (Error) from schemas/outbound/liveness.py. Each event has an `id` "
                                "to send back as Last-Event-ID when reconnecting."}}


@router.get("/{upload_id}/events", response_class=StreamingResponse,
            responses={**SSE_DOC, **error_responses(400, 401)})
async def liveness_events(upload_id: str, headers: Annotated[EventStreamHeaders, Header()]):
    require_api_key(headers.x_api_key)
    require_upload_id(upload_id)
    events = event_stream.get(upload_id)
    start = 0 if headers.last_event_id is None else headers.last_event_id + 1

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
                await asyncio.wait_for(events.changed.wait(), get_settings().keepalive_seconds)
            except TimeoutError:
                yield ": keep-alive\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
