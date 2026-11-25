"""tusd's HTTP hooks (tusd -hooks-http .../hooks/tus). Keep this route reachable only from tusd."""
from fastapi import APIRouter, Request

from liveness_service.core.security import SESSION_ID, UPLOAD_ID
from liveness_service.exceptions import SessionError
from liveness_service.schemas.inbound.tusd_hook import TusdHook
from liveness_service.schemas.outbound.tusd_hook import HookHTTPResponse, TusdHookResponse

router = APIRouter(tags=["tusd"])


@router.post("/hooks/tus", response_model=TusdHookResponse, response_model_exclude_none=True)
async def tus_hook(hook: TusdHook, request: Request):
    uploads = request.app.state.uploads
    upload = hook.Event.Upload

    if hook.Type == "pre-create":  # the only hook that can reject an upload
        try:
            await uploads.authorize_upload(upload.MetaData)
        except SessionError as err:
            return TusdHookResponse(RejectUpload=True,
                                    HTTPResponse=HookHTTPResponse(StatusCode=err.status_code, Body=err.detail))
        return TusdHookResponse()

    # post-* hooks: the metadata was validated at pre-create; ignore anything that doesn't look like ours
    session_id = upload.MetaData.get("session_id", "")
    if not (SESSION_ID.fullmatch(session_id) and UPLOAD_ID.fullmatch(upload.ID or "")):
        return TusdHookResponse()
    if hook.Type == "post-receive":
        await uploads.received(session_id, upload.ID, upload.Offset, upload.Size)
    elif hook.Type == "post-finish":
        await uploads.finished(session_id, upload.ID)
    elif hook.Type == "post-terminate":
        await uploads.terminated(session_id, upload.ID)
    return TusdHookResponse()
