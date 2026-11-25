"""Redis access for sessions, their event channel, the analysis queue and rate limits. Redis is a cache:
anything that must outlive a session goes to the audit store."""
import json
import time

from liveness_service.core.redis import RedisDependent
from liveness_service.enums import SessionStatus

SESSION_KEY = "liveness:session:{}"
EVENTS_CHANNEL = "liveness:events:{}"
JOBS_KEY = "liveness:jobs"
RATE_KEY = "liveness:ratelimit:{}:{}:{}"

# statuses only move forward; a late tusd hook can't move a session back (e.g. post-receive after post-finish)
_ORDER = [SessionStatus.CREATED, SessionStatus.CAPTURING, SessionStatus.UPLOADING, SessionStatus.UPLOADED,
          SessionStatus.ANALYSING, SessionStatus.DONE, SessionStatus.FAILED]
JSON_FIELDS = ("colour_sequence", "display_log", "result")


class SessionRepository(RedisDependent):
    def __init__(self, ttl_seconds):
        self.redis = None  # set by RedisManager.connect() through attach()
        self.ttl = ttl_seconds

    def attach(self, redis_client):
        self.redis = redis_client

    async def create(self, session_id, fields):
        key = SESSION_KEY.format(session_id)
        async with self.redis.pipeline(transaction=True) as pipe:
            await pipe.hset(key, mapping=_encode(fields)).expire(key, self.ttl).execute()

    async def get(self, session_id):
        """The session as a dict (JSON fields decoded), or None if it doesn't exist or has expired."""
        raw = await self.redis.hgetall(SESSION_KEY.format(session_id))
        return _decode(raw) if raw else None

    async def update(self, session_id, **fields):
        """Set fields and refresh the TTL. No-op if the session is gone (never resurrect an expired one)."""
        key = SESSION_KEY.format(session_id)
        # shortcut: exists-then-write isn't atomic; a key expiring in between is re-created without TTL fields
        # only for an instant before EXPIRE. Move to a Lua script if that ever matters.
        if not await self.redis.exists(key):
            return False
        async with self.redis.pipeline(transaction=True) as pipe:
            await pipe.hset(key, mapping=_encode(fields)).expire(key, self.ttl).execute()
        return True

    async def claim(self, session_id, field, value):
        """Set field only if unset (HSETNX). True for the single caller that wins: makes steps single-use.
        False if the session doesn't exist (never create a key without its TTL)."""
        key = SESSION_KEY.format(session_id)
        if not await self.redis.exists(key):
            return False
        won = bool(await self.redis.hsetnx(key, field, _encode_value(value)))
        await self.redis.expire(key, self.ttl)
        return won

    async def advance_status(self, session_id, status, **fields):
        """Move to `status` only if it is later than the current one and the session isn't finished.
        Returns True if it moved."""
        session = await self.get(session_id)
        if not session:
            return False
        current = SessionStatus(session["status"])
        if current.finished or _ORDER.index(current) >= _ORDER.index(status):
            return False
        return await self.update(session_id, status=status, **fields)

    async def publish(self, session_id, event, payload):
        """payload: a schemas.outbound model. Delivered to every SSE connection of the session, on any instance."""
        message = json.dumps({"event": event, "data": payload.model_dump(mode="json", exclude_none=True)})
        await self.redis.publish(EVENTS_CHANNEL.format(session_id), message)

    def subscribe(self, session_id):
        """Pub/Sub for the session's events; use as `async with repo.subscribe(id) as pubsub`."""
        return _Subscription(self.redis, EVENTS_CHANNEL.format(session_id))

    async def enqueue_analysis(self, job):
        await self.redis.lpush(JOBS_KEY, json.dumps(job))

    async def next_analysis(self, timeout_seconds):
        """Block up to timeout for the next job; None on timeout.
        shortcut: BRPOP drops a job if the worker dies mid-analysis; use BLMOVE + a processing list (or RQ)
        for at-least-once delivery."""
        item = await self.redis.brpop([JOBS_KEY], timeout=timeout_seconds)
        return json.loads(item[1]) if item else None

    async def count_hit(self, kind, value, window_seconds):
        """Count one hit for (kind, value) in the current fixed window.
        Returns (hits so far in this window, seconds until the window resets)."""
        now = int(time.time())
        window = now // window_seconds
        key = RATE_KEY.format(kind, value, window)
        async with self.redis.pipeline(transaction=True) as pipe:
            count, _ = await pipe.incr(key).expire(key, window_seconds).execute()
        return count, (window + 1) * window_seconds - now


class _Subscription:
    def __init__(self, redis, channel):
        self.pubsub = redis.pubsub()
        self.channel = channel

    async def __aenter__(self):
        await self.pubsub.subscribe(self.channel)
        return self

    async def __aexit__(self, *exc):
        await self.pubsub.unsubscribe(self.channel)
        await self.pubsub.aclose()

    async def next_message(self, timeout_seconds):
        """{"event", "data"} or None if nothing arrived within the timeout."""
        message = await self.pubsub.get_message(ignore_subscribe_messages=True, timeout=timeout_seconds)
        return json.loads(message["data"]) if message else None


def _encode_value(value):
    return json.dumps(value) if isinstance(value, (dict, list)) else str(value)


def _encode(fields):
    return {k: _encode_value(v) for k, v in fields.items() if v is not None}


def _decode(raw):
    return {k: json.loads(v) if k in JSON_FIELDS else v for k, v in raw.items()}
