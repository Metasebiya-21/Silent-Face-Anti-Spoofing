"""tus endpoints the app uploads to. Each request is forwarded to tusd."""
from fastapi import APIRouter, Request

from liveness_service.core import tusd_client
from liveness_service.core.security import require_api_key, require_upload_id

router = APIRouter(prefix="/v1/liveness", tags=["uploads"])


async def _forward(request, path):
    require_api_key(request.headers.get("x-api-key"))
    return await tusd_client.forward(request, path,
                                     lambda upload_id: str(request.url_for("tus_upload", upload_id=upload_id)))


@router.api_route("", methods=["POST", "OPTIONS"])
async def tus_create(request: Request):
    """tus create. The Location header is /v1/liveness/{upload_id}."""
    return await _forward(request, "")


@router.api_route("/{upload_id}", methods=["HEAD", "PATCH", "DELETE"], name="tus_upload")
async def tus_upload(upload_id: str, request: Request):
    """tus PATCH (append a chunk), HEAD (offset to resume from), DELETE (cancel)."""
    require_upload_id(upload_id)
    return await _forward(request, upload_id)
