import re
import secrets

from fastapi import HTTPException

from liveness_service.core import config

UPLOAD_ID = re.compile(r"[A-Za-z0-9_-]{1,128}")  # tusd filestore IDs are hex; this also blocks path tricks


def authorized(api_key):
    """True if LIVENESS_API_KEY is unset (auth disabled) or api_key matches it."""
    expected = config.api_key()
    return not expected or secrets.compare_digest(api_key or "", expected)


def require_api_key(api_key):
    if not authorized(api_key):
        raise HTTPException(401, "invalid or missing X-API-Key")


def require_upload_id(upload_id):
    if not UPLOAD_ID.fullmatch(upload_id):
        raise HTTPException(400, "invalid upload id")
