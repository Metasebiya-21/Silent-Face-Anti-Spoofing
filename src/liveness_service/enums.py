from enum import StrEnum


class SessionStatus(StrEnum):
    CREATED = "created"        # session issued, nothing happened yet
    CAPTURING = "capturing"    # SSE connected, colour sequence being revealed
    UPLOADING = "uploading"    # tus upload accepted and receiving data
    UPLOADED = "uploaded"      # upload complete, analysis queued
    ANALYSING = "analysing"    # worker picked the job
    DONE = "done"              # decision made
    FAILED = "failed"          # upload terminated or analysis crashed
    EXPIRED = "expired"        # TTL passed (reported from the token; the Redis key is gone by then)

    @property
    def finished(self):
        return self in (SessionStatus.DONE, SessionStatus.FAILED, SessionStatus.EXPIRED)


class Decision(StrEnum):
    """The only outcome the client ever sees."""
    LIVE = "live"
    SPOOF = "spoof"
    RETRY = "retry"


class CheckOutcome(StrEnum):
    """Outcome of one check (here: the colour challenge), before any fusion."""
    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"

    def to_decision(self):
        return {CheckOutcome.PASS: Decision.LIVE, CheckOutcome.FAIL: Decision.SPOOF,
                CheckOutcome.INCONCLUSIVE: Decision.RETRY}[self]
