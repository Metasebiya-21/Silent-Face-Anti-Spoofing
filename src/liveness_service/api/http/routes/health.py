from fastapi import APIRouter

from liveness_service.schemas.outbound.http import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")
