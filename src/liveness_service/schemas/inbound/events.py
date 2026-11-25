"""Request headers of GET /v1/liveness/{upload_id}/events."""
from pydantic import BaseModel, NonNegativeInt


class EventStreamHeaders(BaseModel):
    x_api_key: str | None = None
    last_event_id: NonNegativeInt | None = None  # sent by a reconnecting SSE client; replay starts after it
