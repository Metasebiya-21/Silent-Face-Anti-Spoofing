"""Service settings, validated by pydantic-settings.

Priority, highest first: constructor args, HashiCorp Vault (KV v2), environment variables, .env.
Vault is used when VAULT_ADDR / VAULT_TOKEN / VAULT_PATH are set (env or .env); its secret's keys are
field names, e.g. {"liveness_api_key": "..."}. Locally, leave Vault unset and use .env (see .env.example).
"""
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from typing import Annotated

import hvac
from pydantic import (AnyHttpUrl, Field, NonNegativeInt, PositiveFloat, PositiveInt, SecretStr, TypeAdapter,
                      model_validator)
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

logger = logging.getLogger(__name__)

DEV_SESSION_SECRET = "dev-only-session-secret"  # rejected when production=true
Channel = Annotated[int, Field(ge=0, le=255)]


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
    keepalive_seconds: PositiveInt = 15          # SSE comment interval, so proxies don't close an idle stream
    redis_url: SecretStr = SecretStr("redis://localhost:6379/0")  # secret: usually carries a password

    # sessions
    session_secret: SecretStr = SecretStr(DEV_SESSION_SECRET)  # HMAC key for session tokens; set it in production
    session_ttl_seconds: PositiveInt = 600       # Redis TTL of a session, refreshed while it is active
    rate_limit_max_sessions: PositiveInt = 10    # sessions per device id and per IP ...
    rate_limit_window_seconds: PositiveInt = 60  # ... per this window

    # colour challenge: what the phone shows
    colour_sequence_length: PositiveInt = 10
    colour_palette: dict[str, tuple[Channel, Channel, Channel]] = {  # name -> screen RGB; no saturated pure red
        "white": (255, 255, 255), "blue": (0, 0, 255), "green": (0, 255, 0),
        "yellow": (255, 255, 0), "magenta": (255, 0, 255), "cyan": (0, 255, 255),
    }
    colour_step_min_ms: PositiveInt = 300
    colour_step_max_ms: PositiveInt = 600
    colour_max_changes_per_second: PositiveInt = 3
    colour_reveal_lead_ms: NonNegativeInt = 150  # SSE sends each step this long before it is due
    capture_lead_in_ms: NonNegativeInt = 1000     # time between "start recording" and the first colour

    # colour challenge: verification (worker)
    yunet_model_path: Path = Path("./resources/face_detection/face_detection_yunet_2023mar.onnx")
    yunet_score_threshold: float = Field(0.7, gt=0, lt=1)
    verifier_resample_hz: PositiveInt = 30
    verifier_max_lag_ms: NonNegativeInt = 300
    verifier_step_window_ms: PositiveInt = 150  # averaged before/after each colour change for the step match
    verifier_min_face_frame_ratio: float = Field(0.8, gt=0, le=1)  # fewer frames with a face -> inconclusive
    verifier_min_response_amplitude: PositiveFloat = 0.002  # face chroma std below this -> inconclusive
    verifier_pass_min_correlation: float = Field(0.5, gt=0, le=1)
    verifier_pass_min_step_match: float = Field(0.6, gt=0, le=1)
    verifier_pass_min_face_bg_ratio: PositiveFloat = 1.5
    audit_db_path: Path = Path("./var/audit.sqlite3")

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
    def require_secrets_in_production(self):
        if self.production and not self.liveness_api_key:
            raise ValueError("liveness_api_key must be set in production")
        if self.production and self.session_secret.get_secret_value() == DEV_SESSION_SECRET:
            raise ValueError("session_secret must be set in production")
        return self

    @model_validator(mode="after")
    def colour_rules_are_satisfiable(self):
        if len(self.colour_palette) < 2:
            raise ValueError("colour_palette needs at least 2 colours (no colour may repeat back to back)")
        if self.colour_step_min_ms > self.colour_step_max_ms:
            raise ValueError("colour_step_min_ms must be <= colour_step_max_ms")
        # k consecutive steps must last > 1 s; after k-1 shortest steps, the longest step must still get there
        k = self.colour_max_changes_per_second
        if (k - 1) * self.colour_step_min_ms + self.colour_step_max_ms <= 1000:
            raise ValueError("colour step durations can't satisfy colour_max_changes_per_second")
        return self


@lru_cache
def get_settings():
    """Loaded once per process; restart the service after changing a secret in Vault."""
    return Settings()
