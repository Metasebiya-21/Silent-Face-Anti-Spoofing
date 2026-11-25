"""Service settings, all from environment variables. Read on each call so tests and restarts pick up changes."""
import os
from pathlib import Path

SESSION_TTL_SECONDS = 3600    # how long an upload's events are kept in memory
KEEPALIVE_SECONDS = 15        # SSE comment interval, so proxies don't close an idle stream


def api_key():
    """Shared secret the app sends as X-API-Key; None disables auth."""
    return os.environ.get("LIVENESS_API_KEY")


def tusd_url():
    """tusd's upload endpoint, e.g. http://localhost:8021/files/."""
    return os.environ.get("TUSD_URL", "http://localhost:8080/files/")


def tusd_data_dir():
    """The folder tusd's filestore writes uploads to (its -upload-dir, as seen from this service)."""
    return Path(os.environ.get("TUSD_DATA_DIR", "./data"))
