"""Errors the service raises on purpose, so callers can tell them apart from bugs.

Session errors carry the HTTP status the API answers with; services stay free of FastAPI.
"""


class VideoUnreadableError(ValueError):
    """The uploaded file could not be opened as a video."""


class SessionError(Exception):
    status_code = 400

    def __init__(self, detail):
        super().__init__(detail)
        self.detail = detail


class InvalidToken(SessionError):
    status_code = 401


class SessionNotFound(SessionError):
    status_code = 404


class SessionConflict(SessionError):
    """The session is in the wrong state for this action, e.g. a second upload on a single-use session."""
    status_code = 409


class SessionGone(SessionError):
    """Expired or finished."""
    status_code = 410


class RateLimited(SessionError):
    status_code = 429

    def __init__(self, detail, retry_after_seconds):
        super().__init__(detail)
        self.retry_after_seconds = retry_after_seconds
