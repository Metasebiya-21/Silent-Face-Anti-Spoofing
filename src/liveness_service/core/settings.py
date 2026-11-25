"""Service settings, validated by pydantic-settings.

Priority, highest first: constructor args, HashiCorp Vault (KV v2), environment variables, .env.
Vault is used when VAULT_ADDR / VAULT_TOKEN / VAULT_PATH are set (env or .env); its secret's keys are
field names, e.g. {"liveness_api_key": "..."}. Locally, leave Vault unset and use .env (see .env.example).
"""
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import hvac
from pydantic import AnyHttpUrl, PositiveInt, SecretStr, TypeAdapter, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

logger = logging.getLogger(__name__)


class VaultSettingsSource(PydanticBaseSettingsSource):
    """Reads one secret from HashiCorp Vault KV v2. In production a Vault failure stops startup."""

    def __init__(self, settings_cls: type[BaseSettings], config: Mapping[str, Any]):
        super().__init__(settings_cls)
        self._config = config

    def get_field_value(self, field, field_name):
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        addr, token, path = (self._config.get(k) for k in ("vault_addr", "vault_token", "vault_path"))
        production = self._config["production"]

        if not (addr and token and path):
            if production and self._config["vault_configured"]:
                raise RuntimeError("VAULT_ADDR, VAULT_TOKEN and VAULT_PATH must all be set in production")
            return {}

        try:
            client = hvac.Client(url=addr, token=token, timeout=5)
            return client.secrets.kv.v2.read_secret_version(path=path).get("data", {}).get("data", {})
        except Exception as err:
            message = "Failed to load secrets from HashiCorp Vault"
            if production:
                raise RuntimeError(message) from err
            logger.warning("%s: %s", message, err)
            return {}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Vault connection (environment / .env only)
    vault_addr: str = "http://127.0.0.1:8200"
    vault_token: SecretStr | None = None
    vault_path: str = "liveness-service/config"
    production: bool = False

    liveness_api_key: SecretStr | None = None    # shared secret the app sends as X-API-Key; unset disables auth
    tusd_url: AnyHttpUrl = AnyHttpUrl("http://localhost:8080/files/")  # tusd's upload endpoint
    tusd_data_dir: Path = Path("./data")         # tusd's -upload-dir, as seen from this service
    session_ttl_seconds: PositiveInt = 3600      # how long an upload's events are kept in memory
    keepalive_seconds: PositiveInt = 15          # SSE comment interval, so proxies don't close an idle stream

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings, dotenv_settings,
                                   file_secret_settings):
        # Vault's own connection settings come from the other sources; later dicts win
        values = {**dotenv_settings(), **env_settings(), **init_settings()}
        vault_keys = ("vault_addr", "vault_token", "vault_path")
        vault_config = {
            "vault_addr": values.get("vault_addr") or cls.model_fields["vault_addr"].default,
            "vault_token": values.get("vault_token"),
            "vault_path": values.get("vault_path") or cls.model_fields["vault_path"].default,
            # env values are raw strings here: parse, or "false" would count as true
            "production": TypeAdapter(bool).validate_python(values.get("production", False)),
            "vault_configured": any(k in values for k in vault_keys),
        }
        return init_settings, VaultSettingsSource(settings_cls, vault_config), env_settings, dotenv_settings, \
            file_secret_settings

    @model_validator(mode="after")
    def require_api_key_in_production(self):
        if self.production and not self.liveness_api_key:
            raise ValueError("liveness_api_key must be set in production")
        return self


@lru_cache
def get_settings():
    """Loaded once per process; restart the service after changing a secret in Vault."""
    return Settings()
