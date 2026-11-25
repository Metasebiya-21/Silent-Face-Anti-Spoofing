import pytest
from pydantic import ValidationError

from liveness_service.core import settings as settings_module
from liveness_service.core.settings import Settings


class FakeVault:
    """Stands in for hvac.Client; serves `secret` or raises `error`."""
    secret, error, calls = {}, None, []

    def __init__(self, url, token, timeout):
        FakeVault.calls.append((url, token))
        self.secrets = self
        self.kv = self
        self.v2 = self

    def read_secret_version(self, path):
        if FakeVault.error:
            raise FakeVault.error
        return {"data": {"data": FakeVault.secret}}


@pytest.fixture
def vault(monkeypatch):
    FakeVault.secret, FakeVault.error, FakeVault.calls = {}, None, []
    monkeypatch.setattr(settings_module.hvac, "Client", FakeVault)
    return FakeVault


def test_defaults_without_vault(vault):
    s = Settings()
    assert s.liveness_api_key is None and str(s.tusd_url) == "http://localhost:8080/files/"
    assert vault.calls == []  # no token, so Vault is never contacted


def test_vault_overrides_environment(vault, monkeypatch):
    vault.secret = {"liveness_api_key": "from-vault", "tusd_url": "http://tusd:8080/files/"}
    monkeypatch.setenv("VAULT_TOKEN", "t")
    monkeypatch.setenv("LIVENESS_API_KEY", "from-env")
    s = Settings()
    assert s.liveness_api_key.get_secret_value() == "from-vault"
    assert str(s.tusd_url) == "http://tusd:8080/files/"
    assert vault.calls == [("http://127.0.0.1:8200", "t")]


def test_env_file_is_read_when_vault_unset(tmp_path, monkeypatch, vault):
    (tmp_path / ".env").write_text("LIVENESS_API_KEY=from-dotenv\nKEEPALIVE_SECONDS=5\n")
    monkeypatch.setitem(Settings.model_config, "env_file", tmp_path / ".env")
    s = Settings()
    assert s.liveness_api_key.get_secret_value() == "from-dotenv" and s.keepalive_seconds == 5


def test_vault_failure_is_a_warning_locally_and_fatal_in_production(vault, monkeypatch):
    vault.error = ConnectionError("vault down")
    monkeypatch.setenv("VAULT_TOKEN", "t")
    monkeypatch.setenv("LIVENESS_API_KEY", "k")
    assert Settings().liveness_api_key.get_secret_value() == "k"  # falls back to env

    monkeypatch.setenv("PRODUCTION", "true")
    with pytest.raises(RuntimeError, match="HashiCorp Vault"):
        Settings()


def test_production_flag_string_false_is_false(vault, monkeypatch):
    monkeypatch.setenv("PRODUCTION", "false")
    monkeypatch.setenv("VAULT_ADDR", "http://vault:8200")  # partly configured: would be fatal in production
    assert Settings().production is False


def test_production_requires_complete_vault_config_and_api_key(vault, monkeypatch):
    monkeypatch.setenv("PRODUCTION", "true")
    monkeypatch.setenv("VAULT_ADDR", "http://vault:8200")
    monkeypatch.setenv("VAULT_PATH", "")
    with pytest.raises(RuntimeError, match="must all be set"):
        Settings()

    monkeypatch.delenv("VAULT_ADDR")
    monkeypatch.delenv("VAULT_PATH")
    with pytest.raises(ValidationError, match="liveness_api_key must be set"):
        Settings()


def test_invalid_values_are_rejected(vault, monkeypatch):
    monkeypatch.setenv("TUSD_URL", "not a url")
    with pytest.raises(ValidationError):
        Settings()
