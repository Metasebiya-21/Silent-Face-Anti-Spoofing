"""tus endpoints the app uploads to. Each request is forwarded to tusd."""
from fastapi import APIRouter, Request

from liveness_service.core import tusd_client
from liveness_service.core.security import require_api_key, require_upload_id
from liveness_service.schemas.outbound.http import error_responses

router = APIRouter(prefix="/v1/liveness/uploads", tags=["uploads"])


async def _forward(request, path):
    require_api_key(request.headers.get("x-api-key"))
    return await tusd_client.forward(request, path,
                                     lambda upload_id: str(request.url_for("tus_upload", upload_id=upload_id)))


@router.post("", operation_id="tus_create", responses=error_responses(401, 502))
@router.options("", operation_id="tus_options", responses=error_responses(401, 502))
async def tus_create(request: Request):
    """tus create (the session's tus_endpoint). Location is /v1/liveness/uploads/{upload_id}."""
    return await _forward(request, "")


@router.patch("/{upload_id}", name="tus_upload", operation_id="tus_append", responses=error_responses(400, 401, 502))
@router.head("/{upload_id}", operation_id="tus_offset", responses=error_responses(400, 401, 502))
@router.delete("/{upload_id}", operation_id="tus_cancel", responses=error_responses(400, 401, 502))
async def tus_upload(upload_id: str, request: Request):
    """tus PATCH (append a chunk), HEAD (offset to resume from), DELETE (cancel)."""
    require_upload_id(upload_id)
    return await _forward(request, upload_id)
