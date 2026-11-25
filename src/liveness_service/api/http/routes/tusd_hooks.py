"""Receives tusd's HTTP hooks. Keep this route reachable only from tusd (internal network)."""
from fastapi import APIRouter, Request

from liveness_service.core.security import UPLOAD_ID, authorized
from liveness_service.schemas.inbound.tusd_hook import TusdHook
from liveness_service.schemas.outbound.tusd_hook import HookHTTPResponse, TusdHookResponse
from liveness_service.services import video_liveness

router = APIRouter(tags=["tusd"])


@router.post("/tusd/hooks", response_model=TusdHookResponse, response_model_exclude_none=True)
async def tusd_hook(hook: TusdHook, request: Request):
    upload = hook.Event.Upload

    if hook.Type == "pre-create":  # the only hook that can reject an upload
        if not authorized(hook.header("X-Api-Key")):
            return TusdHookResponse(RejectUpload=True,
                                    HTTPResponse=HookHTTPResponse(StatusCode=401, Body="invalid or missing X-API-Key"))
        return TusdHookResponse()

    if not UPLOAD_ID.fullmatch(upload.ID or ""):
        return TusdHookResponse()
    if hook.Type == "post-receive":
        video_liveness.upload_received(upload.ID, upload.Offset, upload.Size)
    elif hook.Type == "post-finish":
        video_liveness.upload_finished(upload.ID, request.app.state.checker, request.app.state.tasks)
    elif hook.Type == "post-terminate":
        video_liveness.upload_terminated(upload.ID)
    return TusdHookResponse()
