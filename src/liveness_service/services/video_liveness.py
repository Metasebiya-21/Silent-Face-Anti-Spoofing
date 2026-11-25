"""Turns tusd upload stages into events, and runs the liveness check once an upload is complete."""
import asyncio
import logging
from dataclasses import asdict

from liveness_service.core.settings import get_settings
from liveness_service.exceptions import VideoUnreadableError
from liveness_service.schemas.outbound.liveness import Error, Processing, Progress, Result
from liveness_service.services import event_stream

log = logging.getLogger(__name__)


def upload_received(upload_id, offset, size):
    event_stream.get(upload_id).publish("progress", Progress(offset=offset, size=size))


def upload_terminated(upload_id):
    event_stream.get(upload_id).publish("error", Error(detail="upload terminated"), final=True)


def upload_finished(upload_id, checker, tasks):
    """Start the check in the background; tasks keeps a reference so the task isn't garbage-collected."""
    events = event_stream.get(upload_id)
    events.publish("processing", Processing())
    tasks.add(task := asyncio.create_task(_check(upload_id, checker, events)))
    task.add_done_callback(tasks.discard)


async def _check(upload_id, checker, events):
    # path built from our config, not the hook's Storage.Path, so a forged hook can't point us at another file
    path = get_settings().tusd_data_dir / upload_id
    try:
        result = await asyncio.to_thread(checker.check_video, path)
    except VideoUnreadableError as err:
        events.publish("error", Error(detail=str(err)), final=True)
        return
    except Exception:
        log.exception("liveness check failed for upload %s", upload_id)
        events.publish("error", Error(detail="liveness check failed"), final=True)
        return
    payload = Result(face_found=False) if result is None else Result(face_found=True, **asdict(result))
    events.publish("result", payload, final=True)
