from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def _get_bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _get_str_env(name: str, default: str = "") -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip()


def _is_dev_environment() -> bool:
    environment = _get_str_env("APP_ENV", default="development").lower()
    return environment in {"dev", "development", "local"}


def is_evaluation_v3_enabled() -> bool:
    return _get_bool_env("ENABLE_EVALUATION_V3", default=_is_dev_environment())


def get_evaluation_v3_mode() -> str:
    return _get_str_env("EVALUATION_V3_MODE", default="experiment").lower()


def is_evaluation_v3_experiment_mode() -> bool:
    return get_evaluation_v3_mode() == "experiment"
