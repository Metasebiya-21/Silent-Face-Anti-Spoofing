
from abc import ABC, abstractmethod


from redis.asyncio import ConnectionPool, Redis

from liveness_service.core.logger import LoggerProvider
from liveness_service.core.settings import Settings


class RedisDependent(ABC):
    @abstractmethod
    def attach(self, redis_client: Redis) -> None:
        pass


class RedisManager:
    def __init__(self, logger_provider: LoggerProvider, settings: Settings):
        self._logger = logger_provider.get_logger()
        self._redis_dependents: set[RedisDependent] = set()
        self._settings = settings

    async def connect(self) -> None:
        # the URL is a SecretStr (it carries the password); decode_responses: str back from Redis, not bytes
        self._redis_connection_pool = ConnectionPool.from_url(self._settings.redis_url.get_secret_value(),
                                                              decode_responses=True)
        self._redis_client = Redis.from_pool(self._redis_connection_pool)

        await self._redis_client.ping()

        for redis_dependent in self._redis_dependents:
            redis_dependent.attach(self._redis_client)

        self._logger.info("connected to redis")

    async def disconnect(self) -> None:
        await self._redis_connection_pool.aclose()
        self._logger.info("disconnected from redis")

    def add(self, redis_dependent: RedisDependent) -> "RedisManager":
        self._redis_dependents.add(redis_dependent)

        return self
