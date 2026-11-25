"""SSE stream of one session: capture start, each colour just in time, then upload/analysis status and `done`."""
import json

from fastapi import APIRouter, Header, Request
from fastapi.responses import StreamingResponse

from liveness_service.core.security import bearer_token
from liveness_service.schemas.outbound.http import error_responses

router = APIRouter(prefix="/v1/liveness/sessions", tags=["events"])

SSE_DOC = {200: {"content": {"text/event-stream": {}},
                 "description": "Events in order: `capture` (Capture), `colour` (ColourStepEvent) x N, "
                                "`upload_progress` (UploadProgress), `uploaded`, `analysing`, `done` (Done). "
                                "Payloads: schemas/outbound/liveness.py."}}


@router.get("/{session_id}/events", response_class=StreamingResponse,
            responses={**SSE_DOC, **error_responses(401, 410)})
async def session_events(session_id: str, request: Request, authorization: str | None = Header(None)):
    # access errors (401 / 410) raise here, before the 200 and the stream start
    events = await request.app.state.sessions.open_events(session_id, bearer_token(authorization))

    async def stream():
        async for item in events:
            if item is None:
                yield ": keep-alive\n\n"
                continue
            name, payload = item
            data = payload if isinstance(payload, dict) else payload.model_dump(mode="json", exclude_none=True)
            yield f"event: {name}\ndata: {json.dumps(data)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
