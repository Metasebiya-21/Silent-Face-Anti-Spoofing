import base64
import hashlib
import hmac
import re
import secrets
import time

from fastapi import HTTPException

from liveness_service.core.settings import get_settings
from liveness_service.exceptions import InvalidToken, SessionGone

UPLOAD_ID = re.compile(r"[A-Za-z0-9_-]{1,128}")  # tusd filestore IDs are hex; this also blocks path tricks
SESSION_ID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}")  # UUIDv4


def authorized(api_key):
    """True if LIVENESS_API_KEY is unset (auth disabled) or api_key matches it."""
    secret = get_settings().liveness_api_key
    expected = secret.get_secret_value() if secret else None
    return not expected or secrets.compare_digest(api_key or "", expected)


def require_api_key(api_key):
    if not authorized(api_key):
        raise HTTPException(401, "invalid or missing X-API-Key")


def require_upload_id(upload_id):
    if not UPLOAD_ID.fullmatch(upload_id):
        raise HTTPException(400, "invalid upload id")


# Session token: "<expires_at>.<signature>", signature = HMAC-SHA256(secret, "<session_id>:<expires_at>").
# Bound to one session and expiring with it; the server keeps no token state.

def _signature(session_id, expires_at):
    key = get_settings().session_secret.get_secret_value().encode()
    digest = hmac.new(key, f"{session_id}:{expires_at}".encode(), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def issue_session_token(session_id, expires_at):
    """expires_at: unix seconds (int)."""
    return f"{expires_at}.{_signature(session_id, expires_at)}"


def verify_session_token(session_id, token):
    """Raise InvalidToken unless token was issued for session_id; SessionGone if it has expired."""
    expires_at, _, signature = (token or "").partition(".")
    if not expires_at.isdigit() or not SESSION_ID.fullmatch(session_id or ""):
        raise InvalidToken("invalid session token")
    if not hmac.compare_digest(signature, _signature(session_id, int(expires_at))):
        raise InvalidToken("invalid session token")
    if int(expires_at) <= time.time():
        raise SessionGone("session expired")


def bearer_token(authorization):
    """The token from an `Authorization: Bearer <token>` header value, or None."""
    scheme, _, token = (authorization or "").partition(" ")
    return token if scheme.lower() == "bearer" and token else None
