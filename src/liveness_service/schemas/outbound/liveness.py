"""Responses and SSE event payloads of the liveness session API. The SSE event name is the key in EVENTS."""
from datetime import datetime

from pydantic import BaseModel

from liveness_service.enums import Decision, SessionStatus


class CreateSessionResponse(BaseModel):
    session_id: str
    token: str           # send as `Authorization: Bearer <token>`, and as `token` in tus Upload-Metadata
    tus_endpoint: str    # tus create URL; also send `session_id` in Upload-Metadata
    expires_at: datetime


class SessionStatusResponse(BaseModel):
    status: SessionStatus
    decision: Decision | None = None  # set once status is done


# --- SSE events (in the order the client sees them) ---

class Capture(BaseModel):         # event: capture -> start recording now; first colour follows after lead_in_ms
    lead_in_ms: int
    steps: int                    # how many colour events to expect


class ColourStepEvent(BaseModel):  # event: colour -> put this colour on screen now, then log when it showed
    index: int
    colour: str
    rgb: tuple[int, int, int]
    duration_ms: int


class UploadProgress(BaseModel):  # event: upload_progress
    offset: int
    size: int | None


class Uploaded(BaseModel):        # event: uploaded
    pass


class Analysing(BaseModel):       # event: analysing
    pass


class Done(BaseModel):            # event: done (final) -> decision only, never the reason
    decision: Decision
