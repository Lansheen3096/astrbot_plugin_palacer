"""Shared explicit settings and portable AstrBot paths."""

from __future__ import annotations

import os
from pathlib import Path

PLUGIN_NAME = "astrbot_plugin_palacer"
try:
    from astrbot.core.utils.astrbot_path import get_astrbot_data_path
except ImportError:

    def get_astrbot_data_path():
        return str(Path.cwd() / "data")


DATA = Path(os.environ.get("PALACER_DATA_DIR") or get_astrbot_data_path())
PLUGIN_DATA = DATA / "plugin_data" / PLUGIN_NAME
_settings = {}


def configure(config):
    """Keep one live configuration reference for all sibling modules."""
    global _settings
    _settings = config


def setting(key, default=None):
    return _settings.get(key, default)


def integration_id(role):
    return str(
        (_settings.get("integrations") or {}).get(f"{role}_plugin") or ""
    ).strip()


def integration_dir(role):
    name = integration_id(role)
    if not name:
        return PLUGIN_DATA / "disabled" / role
    if Path(name).name != name or name in {".", ".."}:
        raise ValueError("Integration name must be a plugin directory name")
    return DATA / "plugins" / name


def integration_path(key):
    value = str((_settings.get("integrations") or {}).get(key) or "").strip()
    return DATA / value if value else PLUGIN_DATA / "disabled" / key
