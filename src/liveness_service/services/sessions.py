"""Liveness session lifecycle: create, authorize, status, display log, and the event stream with the
just-in-time colour reveal."""
import logging
import time
import uuid

from liveness_service.core.security import issue_session_token, verify_session_token
from liveness_service.enums import Decision, SessionStatus
from liveness_service.exceptions import RateLimited, SessionConflict, SessionGone
from liveness_service.schemas.outbound.liveness import Capture, ColourStepEvent, Done
from liveness_service.services import colour_sequence

logger = logging.getLogger(__name__)


def now_ms():
    return time.time() * 1000


class SessionService:
    def __init__(self, repository, settings):
        self.repo = repository
        self.settings = settings

    async def create(self, device_id, client_ip):
        """Returns (session_id, token, expires_at unix seconds). Raises RateLimited."""
        s = self.settings
        attempt = 0
        for kind, value in (("device", device_id), ("ip", client_ip)):
            if value:
                count, retry_after = await self.repo.count_hit(kind, value, s.rate_limit_window_seconds)
                if count > s.rate_limit_max_sessions:
                    raise RateLimited("too many liveness sessions, try again later", retry_after)
                attempt = attempt or count  # n-th session of this device (or IP) in the window

        session_id = str(uuid.uuid4())
        created_at = int(time.time())
        expires_at = created_at + s.session_ttl_seconds
        steps = colour_sequence.generate(s.colour_palette, s.colour_sequence_length, s.colour_step_min_ms,
                                         s.colour_step_max_ms, s.colour_max_changes_per_second)
        await self.repo.create(session_id, {
            "status": SessionStatus.CREATED, "created_at": created_at, "expires_at": expires_at,
            "colour_sequence": [step.to_dict() for step in steps], "revealed_index": -1, "attempt": attempt,
        })
        logger.info("liveness session %s created (attempt %s)", session_id, attempt)  # never log the sequence
        return session_id, issue_session_token(session_id, expires_at), expires_at

    async def authorize(self, session_id, token):
        """The session, if token is valid for it. InvalidToken (401) / SessionGone (410, expired)."""
        verify_session_token(session_id, token)
        session = await self.repo.get(session_id)
        if session is None:  # valid token but the Redis key is gone: it expired
            raise SessionGone("session expired")
        return session

    async def status(self, session_id, token):
        session = await self.authorize(session_id, token)
        result = session.get("result") or {}
        return SessionStatus(session["status"]), result.get("decision")

    async def save_display_log(self, session_id, token, log):
        session = await self.authorize(session_id, token)
        status = SessionStatus(session["status"])
        if status.finished:
            raise SessionGone("session finished")
        if status not in (SessionStatus.CAPTURING, SessionStatus.UPLOADING, SessionStatus.UPLOADED):
            raise SessionConflict(f"display log not accepted while {status}")
        indices = [step.index for step in log.steps]
        times = [step.shown_at_ms for step in log.steps]
        if len(set(indices)) != len(indices) or max(indices) > int(session["revealed_index"]):
            raise SessionConflict("display log lists colours that were not revealed")
        if times != sorted(times) or times[0] < log.recording_started_at_ms:
            raise SessionConflict("display log times must be increasing and after the recording started")
        if not await self.repo.claim(session_id, "display_log", log.model_dump()):
            raise SessionConflict("display log already received")

    async def open_events(self, session_id, token):
        """Check access and start the capture (once), then return the event stream.

        Raises before anything is streamed (401 / 410). The colour timeline begins capture_lead_in_ms after the
        first open; reconnecting continues the same timeline.
        """
        session = await self.authorize(session_id, token)
        if SessionStatus(session["status"]).finished:
            raise SessionGone("session finished")
        lead_in = self.settings.capture_lead_in_ms
        if await self.repo.claim(session_id, "capture_started_at_ms", int(now_ms() + lead_in)):
            await self.repo.advance_status(session_id, SessionStatus.CAPTURING)
        return self._events(session_id, colour_sequence.from_dicts(session["colour_sequence"]))

    async def _events(self, session_id, steps):
        """Async iterator of (event name, payload), or None for a keep-alive. Each colour is sent
        colour_reveal_lead_ms before it is due. Status events come from Redis Pub/Sub, so they reach this stream
        whichever API instance handled the tusd hook. Ends after `done`."""
        s = self.settings
        async with self.repo.subscribe(session_id) as subscription:
            session = await self.repo.get(session_id)  # re-read after subscribing so no status change is missed
            if session is None:
                raise SessionGone("session expired")
            if session["status"] == SessionStatus.DONE:
                yield "done", Done(decision=session["result"]["decision"])
                return
            start_ms = float(session["capture_started_at_ms"])
            next_index = int(session["revealed_index"]) + 1
            if next_index < len(steps):
                yield "capture", Capture(lead_in_ms=max(0, int(start_ms - now_ms())), steps=len(steps))

            keepalive_at = now_ms() + s.keepalive_seconds * 1000
            while True:
                reveal_at = (start_ms + steps[next_index].offset_ms - s.colour_reveal_lead_ms
                             if next_index < len(steps) else float("inf"))
                wait_ms = max(0.0, min(reveal_at, keepalive_at) - now_ms())
                message = await subscription.next_message(wait_ms / 1000)
                if message:
                    yield message["event"], message["data"]
                    if message["event"] == "done":
                        return
                    continue
                if now_ms() >= reveal_at:
                    step = steps[next_index]
                    yield "colour", ColourStepEvent(index=step.index, colour=step.colour, rgb=step.rgb,
                                                    duration_ms=step.duration_ms)
                    await self.repo.update(session_id, revealed_index=next_index)
                    next_index += 1
                elif now_ms() >= keepalive_at:
                    yield None
                    keepalive_at = now_ms() + s.keepalive_seconds * 1000

    async def finish(self, session_id, decision, result, status=SessionStatus.DONE):
        """Record the outcome and tell every connected client. result is internal (never sent)."""
        if await self.repo.advance_status(session_id, status, result={**result, "decision": decision}):
            await self.repo.publish(session_id, "done", Done(decision=Decision(decision)))
