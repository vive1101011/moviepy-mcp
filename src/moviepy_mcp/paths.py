"""Resolve every filesystem path under the media workspace or the system temp dir."""

from __future__ import annotations

import os
from pathlib import Path

from moviepy_mcp.config import get_settings, temp_root


def font_dirs() -> list[Path]:
    candidates = [
        Path(os.environ.get("WINDIR", "")) / "Fonts",
        Path.home() / "AppData" / "Local" / "Microsoft" / "Windows" / "Fonts",
        Path("/usr/share/fonts"),
        Path("/usr/local/share/fonts"),
        Path.home() / ".local" / "share" / "fonts",
        Path("/Library/Fonts"),
        Path.home() / "Library" / "Fonts",
    ]
    return [path for path in candidates if path.is_dir()]


def _roots(extra: list[Path] | None = None) -> list[str]:
    roots = [os.path.realpath(get_settings().media_root), temp_root()]
    for path in extra or []:
        roots.append(os.path.realpath(path))
    unique: list[str] = []
    for root in roots:
        if root not in unique:
            unique.append(root)
    return unique


def _under(path: str, roots: list[str]) -> bool:
    resolved = os.path.normcase(os.path.realpath(path))
    for root in roots:
        base = os.path.normcase(root)
        if resolved == base or resolved.startswith(base + os.sep):
            return True
    return False


def _absolute(path: str) -> str:
    """Absolute path, keeping an already-absolute string unchanged."""
    raw = os.path.expanduser(path)
    if os.path.isabs(raw):
        return raw
    return os.path.abspath(os.path.join(str(get_settings().media_root), raw))


def _deny(path: str, purpose: str) -> ValueError:
    root = get_settings().media_root
    return ValueError(
        f"{purpose} '{path}' is outside the media workspace ({root}) "
        "and the system temp directory. Move the file in, or set "
        "MOVIEPY_MCP_MEDIA_ROOT to the folder that should be readable."
    )


def resolve_read(path: str) -> str:
    """Return an absolute path that exists inside a read root (workspace, temp, or fonts)."""
    absolute = _absolute(path)
    if not _under(absolute, _roots(font_dirs())):
        raise _deny(path, "Path")
    if not os.path.isfile(absolute):
        raise ValueError(
            f"File not found: '{absolute}'. Provide a path inside the media workspace."
        )
    return absolute


def resolve_write(path: str) -> str:
    """Return an absolute file path inside a write root, creating its parent directory."""
    absolute = _absolute(path)
    if not _under(absolute, _roots()):
        raise _deny(path, "Output path")
    parent = os.path.dirname(absolute)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return absolute


def resolve_dir(path: str) -> str:
    """Return an absolute directory inside a write root, creating it."""
    absolute = _absolute(path)
    if not _under(absolute, _roots()):
        raise _deny(path, "Directory")
    os.makedirs(absolute, exist_ok=True)
    return absolute


def workspace_dir(*parts: str) -> str:
    """A directory under the media root (uploads, downloads, project state)."""
    return resolve_dir(os.path.join(str(get_settings().media_root), *parts))
