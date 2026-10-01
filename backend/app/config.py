from __future__ import annotations

import os
from dataclasses import dataclass


def _csv(name: str, default: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in os.getenv(name, default).split(",") if item.strip())


def _bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    environment: str
    database_url: str | None
    cors_origins: tuple[str, ...]
    allowed_hosts: tuple[str, ...]
    force_https: bool
    secure_cookies: bool
    csrf_header: str
    metrics_token: str | None

    @classmethod
    def from_env(cls) -> "Settings":
        environment = os.getenv("NEXUSAI_ENV", "development").strip().lower()
        force_https = _bool("NEXUSAI_FORCE_HTTPS", environment == "production")
        cors_origins = _csv(
            "NEXUSAI_CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173",
        )
        allowed_hosts = _csv(
            "NEXUSAI_ALLOWED_HOSTS", "localhost,127.0.0.1,testserver"
        )
        if not cors_origins or "*" in cors_origins:
            raise ValueError("NEXUSAI_CORS_ORIGINS must contain explicit origins")
        if not allowed_hosts:
            raise ValueError("NEXUSAI_ALLOWED_HOSTS must contain at least one host")
        return cls(
            environment=environment,
            database_url=os.getenv("NEXUSAI_DATABASE_URL") or None,
            cors_origins=cors_origins,
            allowed_hosts=allowed_hosts,
            force_https=force_https,
            secure_cookies=_bool("NEXUSAI_SECURE_COOKIES", force_https),
            csrf_header=os.getenv("NEXUSAI_CSRF_HEADER", "X-NexusAI-CSRF"),
            metrics_token=os.getenv("NEXUSAI_METRICS_TOKEN") or None,
        )


settings = Settings.from_env()
