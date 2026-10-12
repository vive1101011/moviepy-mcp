"""Process settings. Read from the environment on each call so tests can override them."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


def _flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    return int(raw)


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    return float(raw)


@dataclass(frozen=True)
class Settings:
    media_root: Path
    advanced: bool
    persist: bool
    isolate_sessions: bool
    allow_browser_cookies: bool
    max_clips: int
    max_width: int
    max_height: int
    max_duration_seconds: float
    max_sessions: int
    transport: str
    host: str
    port: int
    token: str | None
    allowed_hosts: list[str]


def get_settings() -> Settings:
    root = os.environ.get("MOVIEPY_MCP_MEDIA_ROOT")
    media_root = Path(root).expanduser() if root else Path.home()
    hosts = [
        item.strip()
        for item in os.environ.get("MOVIEPY_MCP_ALLOWED_HOSTS", "").split(",")
        if item.strip()
    ]
    token = os.environ.get("MOVIEPY_MCP_TOKEN") or None
    return Settings(
        media_root=media_root,
        advanced=_flag("MOVIEPY_MCP_ADVANCED", False),
        persist=_flag("MOVIEPY_MCP_PERSIST", True),
        isolate_sessions=_flag("MOVIEPY_MCP_ISOLATE_SESSIONS", False),
        allow_browser_cookies=_flag("MOVIEPY_MCP_ALLOW_BROWSER_COOKIES", False),
        max_clips=_int("MOVIEPY_MCP_MAX_CLIPS", 48),
        max_width=_int("MOVIEPY_MCP_MAX_WIDTH", 4096),
        max_height=_int("MOVIEPY_MCP_MAX_HEIGHT", 4096),
        max_duration_seconds=_float("MOVIEPY_MCP_MAX_DURATION", 7200),
        max_sessions=_int("MOVIEPY_MCP_MAX_SESSIONS", 32),
        transport=os.environ.get("MOVIEPY_MCP_TRANSPORT", "stdio").strip().lower(),
        host=os.environ.get("MOVIEPY_MCP_HOST", "127.0.0.1"),
        port=_int("MOVIEPY_MCP_PORT", 8000),
        token=token,
        allowed_hosts=hosts,
    )


def temp_root() -> str:
    return os.path.realpath(tempfile.gettempdir())
