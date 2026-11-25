"""One analysis job: run the colour-challenge verifier on an uploaded capture, decide, audit, notify."""
import asyncio
import logging
import time

from liveness_service.enums import Decision, SessionStatus
from liveness_service.exceptions import VideoUnreadableError
from liveness_service.ml.colour_challenge.signals import ShownStep, VerifierConfig
from liveness_service.ml.colour_challenge.verifier import verify_video
from liveness_service.repositories.audit_repository import AuditRecord
from liveness_service.schemas.outbound.liveness import Analysing

logger = logging.getLogger(__name__)


def verifier_config(settings):
    return VerifierConfig(
        resample_hz=settings.verifier_resample_hz, max_lag_ms=settings.verifier_max_lag_ms,
        step_window_ms=settings.verifier_step_window_ms,
        min_face_frame_ratio=settings.verifier_min_face_frame_ratio,
        min_response_amplitude=settings.verifier_min_response_amplitude,
        pass_min_correlation=settings.verifier_pass_min_correlation,
        pass_min_step_match=settings.verifier_pass_min_step_match,
        pass_min_face_bg_ratio=settings.verifier_pass_min_face_bg_ratio,
    )


def shown_steps(session):
    """The colours as actually displayed, from the display log, in recording time (ms since video t=0).
    Colours come from the server's sequence, never from the client."""
    log, sequence = session["display_log"], session["colour_sequence"]
    t0 = log["recording_started_at_ms"]
    steps = sorted(log["steps"], key=lambda s: s["shown_at_ms"])
    return [ShownStep(shown_at_ms=s["shown_at_ms"] - t0, rgb=tuple(sequence[s["index"]]["rgb"])) for s in steps]


class AnalysisService:
    def __init__(self, repository, sessions, audit_store, settings, regions):
        self.repo = repository
        self.sessions = sessions
        self.audit = audit_store
        self.settings = settings
        self.regions = regions  # FaceRegions (YuNet), loaded once per worker

    async def run(self, job):
        session_id, upload_id = job["session_id"], job["upload_id"]
        session = await self.repo.get(session_id)
        if session is None:
            logger.warning("analysis skipped: session %s expired before it ran", session_id)
            return
        started_at = int(time.time())
        if not await self.repo.advance_status(session_id, SessionStatus.ANALYSING, analysis_started_at=started_at):
            return  # already analysed (duplicate job)
        await self.repo.publish(session_id, "analysing", Analysing())

        outcome, scores, error, status = None, {}, None, SessionStatus.DONE
        try:
            if not session.get("display_log"):
                error = "missing display log"
            else:
                path = self.settings.tusd_data_dir / upload_id
                outcome, scores = await asyncio.to_thread(
                    verify_video, path, shown_steps(session), verifier_config(self.settings), self.regions)
        except VideoUnreadableError as err:
            error = str(err)
        except Exception:
            logger.exception("colour-challenge analysis failed for session %s", session_id)
            error, status = "analysis failed", SessionStatus.FAILED

        decision = outcome.to_decision() if outcome else Decision.RETRY
        result = {"outcome": outcome and str(outcome), "scores": scores, "error": error}
        await asyncio.to_thread(self.audit.record, AuditRecord(
            session_id=session_id, decision=decision, outcome=result["outcome"], scores=scores, error=error,
            timings={**{k: session.get(k) for k in ("created_at", "capture_started_at_ms", "uploaded_at")},
                     "analysis_started_at": started_at, "analysed_at": int(time.time())},
        ))
        await self.sessions.finish(session_id, decision, result, status=status)
        logger.info("liveness session %s: %s", session_id, decision)  # decision only; scores are in the audit
