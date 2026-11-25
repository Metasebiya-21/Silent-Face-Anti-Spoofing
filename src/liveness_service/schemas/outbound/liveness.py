"""Payloads of the SSE events sent on /v1/liveness/{upload_id}/events. The SSE event name says which one it is."""
from pydantic import BaseModel


class Progress(BaseModel):      # event: progress
    offset: int
    size: int | None            # None if the app created the upload with a deferred length


class Processing(BaseModel):    # event: processing
    pass


class Result(BaseModel):        # event: result (final)
    face_found: bool
    # only set when face_found is true
    is_real: bool | None = None
    real_ratio: float | None = None
    frames_with_face: int | None = None
    frames_sampled: int | None = None


class Error(BaseModel):         # event: error (final)
    detail: str
