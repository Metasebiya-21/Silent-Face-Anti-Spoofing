import logging


class LoggerProvider:
    """Hands out the service's logger, so components get it injected instead of creating their own."""

    def __init__(self, name: str = "liveness_service"):
        self._name = name

    def get_logger(self) -> logging.Logger:
        return logging.getLogger(self._name)
