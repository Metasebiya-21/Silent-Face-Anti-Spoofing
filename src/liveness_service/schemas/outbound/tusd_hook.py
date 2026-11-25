"""What /tusd/hooks answers tusd (tusd v2 HTTP hooks). Unset fields are left out of the JSON."""
from pydantic import BaseModel


class HookHTTPResponse(BaseModel):
    StatusCode: int
    Body: str | None = None


class TusdHookResponse(BaseModel):
    RejectUpload: bool | None = None          # honored by tusd on pre-create only
    HTTPResponse: HookHTTPResponse | None = None  # what tusd sends the app instead of its normal response
