"""What tusd POSTs to /tusd/hooks (tusd v2 HTTP hooks). Unknown fields are ignored."""
from pydantic import BaseModel, Field


class UploadInfo(BaseModel):
    ID: str | None = None   # null on pre-create: the upload doesn't exist yet
    Size: int | None = None
    Offset: int = 0
    MetaData: dict[str, str] = {}  # the app's tus Upload-Metadata, decoded: session_id, token


class RequestInfo(BaseModel):
    Header: dict[str, list[str]] = {}  # tusd sends each header's values as a list, names Go-canonicalized


class HookEvent(BaseModel):
    Upload: UploadInfo = Field(default_factory=UploadInfo)
    HTTPRequest: RequestInfo = Field(default_factory=RequestInfo)


class TusdHook(BaseModel):
    Type: str                # pre-create, post-receive, post-finish, post-terminate, ...
    Event: HookEvent = Field(default_factory=HookEvent)

    def header(self, name):
        """First value of a request header, or None."""
        return (self.Event.HTTPRequest.Header.get(name) or [None])[0]
