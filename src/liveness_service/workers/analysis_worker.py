"""`liveness-worker`: takes analysis jobs from Redis and runs the colour-challenge verification.

Run several for throughput; each handles one job at a time.
"""
import asyncio
import logging
import signal

from liveness_service.core.logger import LoggerProvider
from liveness_service.core.redis import RedisManager
from liveness_service.core.settings import get_settings
from liveness_service.ml.colour_challenge.regions import FaceRegions
from liveness_service.repositories.audit_repository import SQLiteAuditStore
from liveness_service.repositories.session_repository import SessionRepository
from liveness_service.services.analysis import AnalysisService
from liveness_service.services.sessions import SessionService

logger = logging.getLogger(__name__)
POLL_SECONDS = 5  # how often the loop wakes up to check for shutdown while the queue is empty


async def run():
    settings = get_settings()
    repository = SessionRepository(settings.session_ttl_seconds)
    redis = RedisManager(LoggerProvider(), settings).add(repository)
    await redis.connect()
    analysis = AnalysisService(repository, SessionService(repository, settings),
                               SQLiteAuditStore(settings.audit_db_path), settings,
                               FaceRegions(settings.yunet_model_path, settings.yunet_score_threshold))

    stop = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):  # finish the current job, then exit
        asyncio.get_running_loop().add_signal_handler(sig, stop.set)
    logger.info("liveness worker ready")
    try:
        while not stop.is_set():
            job = await repository.next_analysis(POLL_SECONDS)
            if job:
                await analysis.run(job)
    finally:
        await redis.disconnect()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    asyncio.run(run())
