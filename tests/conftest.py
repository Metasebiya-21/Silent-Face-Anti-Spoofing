import fakeredis
import pytest
from fastapi.testclient import TestClient

from liveness_service.core.redis import RedisManager
from liveness_service.core.settings import Settings, get_settings

SETTINGS_ENV = ("VAULT_ADDR", "VAULT_TOKEN", "VAULT_PATH", "PRODUCTION", "LIVENESS_API_KEY", "REDIS_URL",
                "SESSION_SECRET", "TUSD_URL", "TUSD_DATA_DIR", "AUDIT_DB_PATH")


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch):
    """Tests never read the developer's .env, a real Vault or real Redis, and each starts from fresh settings."""
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    for name in SETTINGS_ENV:
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def setenv(monkeypatch):
    """Set an environment variable and reload settings so the app sees it."""
    def set_(name, value):
        monkeypatch.setenv(name, str(value))
        get_settings.cache_clear()
    return set_


@pytest.fixture
def redis_server():
    return fakeredis.FakeServer()


@pytest.fixture
def redis_inspect(redis_server):
    """Synchronous view of the same fake Redis the app uses, for assertions."""
    return fakeredis.FakeRedis(server=redis_server, decode_responses=True)


@pytest.fixture
def make_client(monkeypatch, redis_server, tmp_path, setenv):
    """Start the app on fake Redis with fast colour timings; extra settings as keyword arguments."""
    async def connect(self):
        self._redis_client = fakeredis.FakeAsyncRedis(server=redis_server, decode_responses=True)
        for dependent in self._redis_dependents:
            dependent.attach(self._redis_client)

    async def disconnect(self):
        await self._redis_client.aclose()

    monkeypatch.setattr(RedisManager, "connect", connect)
    monkeypatch.setattr(RedisManager, "disconnect", disconnect)
    clients = []

    def make(**env):
        defaults = dict(COLOUR_SEQUENCE_LENGTH=3, COLOUR_STEP_MIN_MS=300, COLOUR_STEP_MAX_MS=450,
                        CAPTURE_LEAD_IN_MS=0, COLOUR_REVEAL_LEAD_MS=0, KEEPALIVE_SECONDS=1,
                        SESSION_SECRET="test-secret", TUSD_DATA_DIR=tmp_path / "uploads",
                        AUDIT_DB_PATH=tmp_path / "audit.sqlite3")
        for name, value in {**defaults, **env}.items():
            setenv(name, value)
        from liveness_service.api.http.app import app
        client = TestClient(app)
        client.__enter__()  # runs the lifespan
        clients.append(client)
        return client

    yield make
    for client in clients:
        client.__exit__(None, None, None)


@pytest.fixture
def client(make_client):
    return make_client()
