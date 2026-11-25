import pytest

from liveness_service.core.settings import Settings, get_settings


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch):
    """Tests never read the developer's .env or a real Vault, and each test starts from fresh settings."""
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    for name in ("VAULT_ADDR", "VAULT_TOKEN", "VAULT_PATH", "PRODUCTION", "LIVENESS_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def setenv(monkeypatch):
    """Set an environment variable and reload settings so the app sees it."""
    def set_(name, value):
        monkeypatch.setenv(name, value)
        get_settings.cache_clear()
    return set_
