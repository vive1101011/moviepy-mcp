"""Tool modules. Importing them registers tools on the shared FastMCP app."""

from moviepy_mcp.server import mcp
from moviepy_mcp.tools import audio, composite, media, output, picture, timeline
from moviepy_mcp.tools.media import MEDIA_UPLOAD

mcp.add_provider(MEDIA_UPLOAD)

__all__ = ["audio", "composite", "media", "output", "picture", "timeline"]
