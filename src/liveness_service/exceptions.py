"""Errors the service raises on purpose, so callers can tell them apart from bugs."""


class VideoUnreadableError(ValueError):
    """The uploaded file could not be opened as a video."""
