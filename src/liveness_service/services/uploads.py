"""Session side of tusd's upload hooks: gate the upload, track progress, queue the analysis."""
import logging
import time

from liveness_service.core.settings import get_settings
from liveness_service.enums import Decision, SessionStatus
from liveness_service.exceptions import SessionConflict, SessionGone
from liveness_service.schemas.outbound.liveness import Uploaded, UploadProgress

logger = logging.getLogger(__name__)


class UploadService:
    def __init__(self, repository, sessions):
        self.repo = repository
        self.sessions = sessions

    async def authorize_upload(self, metadata):
        """pre-create: the tus Upload-Metadata must carry a valid session_id + token for a session that hasn't
        uploaded yet. Raises a SessionError (its status code goes back to the app)."""
        session_id = metadata.get("session_id")
        session = await self.sessions.authorize(session_id, metadata.get("token"))
        status = SessionStatus(session["status"])
        if status.finished:
            raise SessionGone("session finished")
        if status not in (SessionStatus.CREATED, SessionStatus.CAPTURING):
            raise SessionConflict("session already has an upload")
        if not await self.repo.claim(session_id, "upload_claimed", 1):  # single-use, even under a race
            raise SessionConflict("session already has an upload")

    async def _owned(self, session_id, upload_id):
        """True if upload_id is this session's upload (the first one seen claims it)."""
        await self.repo.claim(session_id, "tus_upload_id", upload_id)
        session = await self.repo.get(session_id)
        return session is not None and session.get("tus_upload_id") == upload_id

    async def received(self, session_id, upload_id, offset, size):
        if not await self._owned(session_id, upload_id):
            return
        await self.repo.advance_status(session_id, SessionStatus.UPLOADING)
        session = await self.repo.get(session_id)
        if session is None or session["status"] != SessionStatus.UPLOADING:
            return  # tusd sends post-receive asynchronously; one can arrive after post-finish
        await self.repo.update(session_id, upload_offset=offset, upload_size=size)
        await self.repo.publish(session_id, "upload_progress", UploadProgress(offset=offset, size=size))

    async def finished(self, session_id, upload_id):
        if not await self._owned(session_id, upload_id):
            return
        tus_url = f"{str(get_settings().tusd_url).rstrip('/')}/{upload_id}"
        if await self.repo.advance_status(session_id, SessionStatus.UPLOADED, tus_url=tus_url,
                                          uploaded_at=int(time.time())):
            await self.repo.publish(session_id, "uploaded", Uploaded())
            await self.repo.enqueue_analysis({"session_id": session_id, "upload_id": upload_id})

    async def terminated(self, session_id, upload_id):
        if await self._owned(session_id, upload_id):
            await self.sessions.finish(session_id, Decision.RETRY, {"error": "upload terminated"},
                                       status=SessionStatus.FAILED)
