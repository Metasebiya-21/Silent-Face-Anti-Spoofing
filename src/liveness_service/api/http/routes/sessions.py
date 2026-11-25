"""Liveness sessions: one per capture attempt. Every route but create needs the session token."""
from datetime import datetime, timezone

from fastapi import APIRouter, Header, Request, Response

from liveness_service.core.security import bearer_token, require_api_key
from liveness_service.schemas.inbound.sessions import CreateSessionRequest, DisplayLog
from liveness_service.schemas.outbound.http import error_responses
from liveness_service.schemas.outbound.liveness import CreateSessionResponse, SessionStatusResponse

router = APIRouter(prefix="/v1/liveness/sessions", tags=["sessions"])


@router.post("", status_code=201, response_model=CreateSessionResponse, responses=error_responses(401, 429))
async def create_session(request: Request, body: CreateSessionRequest | None = None,
                         x_api_key: str | None = Header(None)):
    require_api_key(x_api_key)
    session_id, token, expires_at = await request.app.state.sessions.create(
        body.device_id if body else None, request.client.host if request.client else None)
    return CreateSessionResponse(session_id=session_id, token=token,
                                 tus_endpoint=str(request.url_for("tus_create")),
                                 expires_at=datetime.fromtimestamp(expires_at, timezone.utc))


@router.get("/{session_id}", response_model=SessionStatusResponse, response_model_exclude_none=True,
            responses=error_responses(401, 410))
async def session_status(session_id: str, request: Request, authorization: str | None = Header(None)):
    status, decision = await request.app.state.sessions.status(session_id, bearer_token(authorization))
    return SessionStatusResponse(status=status, decision=decision)


@router.post("/{session_id}/display-log", status_code=204, responses=error_responses(401, 409, 410))
async def display_log(session_id: str, body: DisplayLog, request: Request, authorization: str | None = Header(None)):
    await request.app.state.sessions.save_display_log(session_id, bearer_token(authorization), body)
    return Response(status_code=204)
