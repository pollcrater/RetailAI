from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv_if_available() -> None:
    """Loads .env if python-dotenv is installed; no-op otherwise."""

    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return

    # Load from repo root by default
    load_dotenv(override=False)


_load_dotenv_if_available()


def _get_env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


@dataclass(frozen=True)
class Settings:
    app_name: str
    app_env: str
    log_level: str
    data_dir: Path


def get_settings() -> Settings:
    data_dir_raw = _get_env("DATA_DIR", "./data") or "./data"

    return Settings(
        app_name=_get_env("APP_NAME", "RetailAI") or "RetailAI",
        app_env=_get_env("APP_ENV", "dev") or "dev",
        log_level=_get_env("LOG_LEVEL", "INFO") or "INFO",
        data_dir=Path(data_dir_raw).resolve(),
    )
