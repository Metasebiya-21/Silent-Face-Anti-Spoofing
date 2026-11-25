"""Per-upload event log that SSE clients read from. Late or reconnecting clients replay what they missed."""
import asyncio

from liveness_service.core.settings import get_settings


class UploadEvents:
    def __init__(self):
        self.events = []  # (event name, data dict); an event's SSE id is its index
        self.done = False
        self.changed = asyncio.Event()

    def publish(self, name, payload, final=False):
        """payload: a schemas.outbound model. After a final event, anything else is dropped."""
        if self.done:  # tusd sends post-receive asynchronously; it can arrive after post-finish's result
            return
        self.events.append((name, payload.model_dump(exclude_none=True)))
        self.done = final
        self.changed.set()
        self.changed = asyncio.Event()  # current waiters woke on the old one; later waiters get a fresh one


# shortcut: in-memory, so run a single uvicorn worker; move to Redis pub/sub to scale out or survive restarts
sessions = {}


def get(upload_id):
    """The upload's event log, created on first use (the SSE client may connect before tusd's first hook)."""
    if upload_id not in sessions:
        sessions[upload_id] = UploadEvents()
        # shortcut: sessions expire session_ttl_seconds after creation, even if the upload is still running
        asyncio.get_running_loop().call_later(get_settings().session_ttl_seconds, sessions.pop, upload_id, None)
    return sessions[upload_id]
