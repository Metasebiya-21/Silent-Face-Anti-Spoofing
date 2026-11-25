from pydantic import BaseModel, Field, NonNegativeInt


class CreateSessionRequest(BaseModel):
    device_id: str | None = Field(None, min_length=1, max_length=128)  # for rate limiting; IP is used too


class DisplayedStep(BaseModel):
    index: NonNegativeInt   # index of the colour event received over SSE
    shown_at_ms: int        # phone clock (unix ms) when the colour actually appeared on screen


class DisplayLog(BaseModel):
    """When each colour was really on screen, on the phone's clock. Post it before the last upload chunk."""
    recording_started_at_ms: int   # phone clock (unix ms) of the recording's first frame (video t = 0)
    steps: list[DisplayedStep] = Field(min_length=1)
