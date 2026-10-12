"""MoviePy MCP server. Tools live in moviepy_mcp.tools."""

from __future__ import annotations

import logging
import os

from fastmcp import FastMCP

from moviepy_mcp.config import get_settings

mcp = FastMCP(
    name="moviepy",
    instructions=(
        "Video and audio editor. Files must sit in the media workspace "
        "(MOVIEPY_MCP_MEDIA_ROOT, default: your home directory) or the system "
        "temp directory. Relative paths are read from the workspace. "
        "Workflow: to add a file from the chat, call file_manager and then "
        "list_files — each upload has a clip_id. Otherwise load_video / "
        "load_audio / load_image, upload_media, or download_video. "
        "Each edit returns a new clip_id — always pass the newest id. "
        "preview_frame before export. To hand the finished clip back in the "
        "chat, call download_media (player and download button). "
        "export_clip writes a path you choose. "
        "export_clip presets: youtube_1080p, reels_1080x1920, podcast_aac. "
        "list_fonts if text rendering fails. project_status after a restart "
        "shows which source clips were reopened. "
        "Blur, chroma key, Ken Burns, and the other effects are hidden unless "
        "MOVIEPY_MCP_ADVANCED=1."
    ),
)


def main() -> None:
    """stdio by default. HTTP when MOVIEPY_MCP_TRANSPORT=http."""
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
    settings = get_settings()
    if settings.transport in ("", "stdio"):
        mcp.run(transport="stdio")
        return
    if settings.transport not in ("http", "streamable-http", "sse"):
        raise SystemExit(
            f"Unknown MOVIEPY_MCP_TRANSPORT '{settings.transport}'. "
            "Use stdio, http, streamable-http, or sse."
        )
    if not settings.token:
        raise SystemExit(
            "HTTP transport requires MOVIEPY_MCP_TOKEN, a bearer token clients send "
            "as Authorization."
        )
    os.environ["MOVIEPY_MCP_ISOLATE_SESSIONS"] = "1"
    from fastmcp.server.auth.providers.jwt import StaticTokenVerifier

    mcp.auth = StaticTokenVerifier({
        settings.token: {"client_id": "moviepy-mcp", "scopes": ["edit"]},
    })
    mcp.run(
        transport=settings.transport,  # type: ignore[arg-type]
        host=settings.host,
        port=settings.port,
        host_origin_protection="auto",
        allowed_hosts=settings.allowed_hosts or None,
    )


from moviepy_mcp.editing import _parse_subtitles, _ytdlp_progress_hook
from moviepy_mcp.registry import _REGISTRY, _register
from moviepy_mcp import prompts as _prompts  # noqa: E402, F401
from moviepy_mcp.tools import (  # noqa: E402, F401
    audio,
    composite,
    media,
    output,
    picture,
    timeline,
)

if not get_settings().advanced:
    mcp.disable(tags={"advanced"})

__all__ = [
    "mcp",
    "main",
    "_REGISTRY",
    "_register",
    "_parse_subtitles",
    "_ytdlp_progress_hook",
]


if __name__ == "__main__":
    main()
