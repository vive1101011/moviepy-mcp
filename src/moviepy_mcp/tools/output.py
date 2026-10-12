"""Frames and rendered files."""

from __future__ import annotations

import asyncio
import base64
import mimetypes
import os
import threading
import uuid
from typing import Any, Optional

from fastmcp import Context
from fastmcp.tools.base import _prefab_to_json
from fastmcp.tools.tool import ToolResult
from mcp.types import TextContent, ToolAnnotations
from prefab_ui.actions import SetState, ShowToast
from prefab_ui.actions.mcp import CallTool
from prefab_ui.app import PrefabApp
from prefab_ui.components import (
    H3,
    Audio,
    Card,
    CardContent,
    CardHeader,
    Column,
    Image,
    Link,
    Muted,
    Small,
    Video,
)
from prefab_ui.components.control_flow import Else, If
from prefab_ui.rx import ERROR, RESULT, STATE

from moviepy_mcp.paths import resolve_read, resolve_write, workspace_dir
from moviepy_mcp.registry import _describe, _get
from moviepy_mcp.runtime import LOCK, serialized
from moviepy_mcp.server import mcp
from moviepy_mcp.editing import (
    _AUDIO_EXTS,
    _STILL_IMAGE_EXTS,
    _bridge_progress,
    _last_frame_time,
    _moviepy_logger,
    _write_clip,
)
from moviepy_mcp.tools.media import MEDIA_UPLOAD

# shortcut: the player uses a data URL, so large files only show their saved path. Stream from an HTTP route when one exists.
PLAYBACK_LIMIT = 16 * 1024 * 1024

READ = ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False)
EDIT = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False)
DESTROY = ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=False)
NET = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=True)


@mcp.tool(tags={'core'}, annotations=DESTROY)
@serialized
def save_frame(clip_id: str, output_path: str,
               time_seconds: float = 0.0, at_end: bool = False) -> dict:
    """Save a single frame of a video as an image (png/jpg).

    Writes a file. Prefer ``preview_frame`` when the model needs to see
    the edit without touching disk.

    Args:
        clip_id: The video or image clip.
        output_path: Destination path.
        time_seconds: Frame time to grab (0.0 = first frame). Ignored if
            at_end=True.
        at_end: Grab the last frame instead of a specific time.
    """
    entry = _get(clip_id, expect=("video", "image"))
    output_path = resolve_write(output_path)
    t = _last_frame_time(entry.clip) if at_end else time_seconds
    entry.clip.save_frame(output_path, t=t)
    return {"saved": output_path, "time_seconds": t, "clip_id": clip_id}

@mcp.tool(tags={'core'}, annotations=DESTROY)
@serialized
def export_image(clip_id: str, output_path: str,
                 time_seconds: float = 0.0, at_end: bool = False) -> dict:
    """Export a still image file from a video or image clip.

    Args:
        clip_id: The video or image clip.
        output_path: Destination path; extension must be .png, .jpg, .jpeg, or .webp.
        time_seconds: Frame time for video clips (ignored for still images,
            and ignored if at_end=True).
        at_end: Grab the last frame instead of a specific time.
    """
    entry = _get(clip_id, expect=("video", "image"))
    ext = os.path.splitext(output_path)[1].lower()
    allowed = {".png", ".jpg", ".jpeg", ".webp"}
    if ext not in allowed:
        raise ValueError(
            f"export_image requires extension in {sorted(allowed)}, got '{ext}'. "
            "Use save_frame for other formats or export_clip for video/audio.")
    output_path = resolve_write(output_path)
    t = _last_frame_time(entry.clip) if at_end else time_seconds
    entry.clip.save_frame(output_path, t=t)
    size = os.path.getsize(output_path)
    return {"exported": output_path, "file_size_bytes": size,
            "time_seconds": t, "clip_id": clip_id,
            "history": entry.history}

@mcp.tool(tags={'core'}, annotations=DESTROY)
@serialized
async def export_clip(clip_id: str, output_path: str, ctx: Context,
                      fps: Optional[float] = None,
                      codec: Optional[str] = None,
                      bitrate: Optional[str] = None,
                      preset: Optional[str] = None) -> dict:
    """Render a clip to disk. THIS is the step that writes a file.

    Args:
        clip_id: The clip to render.
        output_path: Destination path. Extension picks the container:
            .mp4/.webm/.gif for video, .mp3/.wav/.ogg for audio.
        fps: Frames per second for video (defaults to source fps or 24).
        codec: Optional codec override (e.g. 'libx264', 'libvpx').
        bitrate: Optional bitrate (e.g. '4000k').
        preset: Optional delivery preset: youtube_1080p, reels_1080x1920,
            or podcast_aac. Sets size, codec, pixel format, and loudness.
            Explicit fps/codec/bitrate override the preset.

    Returns:
        The written path and final file size.
    """
    entry = _get(clip_id)
    output_path = resolve_write(output_path)

    await ctx.info(f"Exporting {clip_id} to {output_path}")
    await ctx.report_progress(0, None, "starting export")
    report = _bridge_progress(ctx)
    cancel = threading.Event()
    logger = _moviepy_logger(report, cancel)
    loop = asyncio.get_running_loop()

    def work() -> None:
        with LOCK:
            _write_clip(entry, output_path, fps, codec, bitrate, logger, preset)

    try:
        await loop.run_in_executor(None, work)
    except asyncio.CancelledError:
        cancel.set()
        raise
    await ctx.report_progress(1, 1, "export complete")

    size = os.path.getsize(output_path)
    return {"exported": output_path, "file_size_bytes": size,
            "clip_id": clip_id, "history": entry.history}

def _format_bytes(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes / (1024 * 1024):.1f} MB"


def playback_kind(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in _STILL_IMAGE_EXTS or ext == ".gif":
        return "image"
    if ext in _AUDIO_EXTS:
        return "audio"
    return "video"


def playback_payload(path: str, *, limit: int = PLAYBACK_LIMIT) -> dict[str, Any]:
    """Data URL for the download card, or a path-only result when the file is too big."""
    path = resolve_read(path)
    size = os.path.getsize(path)
    kind = playback_kind(path)
    if size > limit:
        return {
            "ready": False,
            "src": "",
            "kind": kind,
            "path": path,
            "file_size_bytes": size,
            "error": (
                f"{_format_bytes(size)} is too large to play in the chat. "
                f"Open the saved file instead."
            ),
        }
    mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
    with open(path, "rb") as fh:
        raw = fh.read()
    src = "data:" + mime + ";base64," + base64.b64encode(raw).decode("ascii")
    return {
        "ready": True,
        "src": src,
        "kind": kind,
        "path": path,
        "file_size_bytes": size,
        "error": "",
    }


def _download_card(path: str, size: int, kind: str) -> PrefabApp:
    playable = size <= PLAYBACK_LIMIT
    with Card(css_class="max-w-2xl mx-auto") as view:
        with CardHeader():
            H3("Ready")
        with CardContent(), Column(gap=3):
            Muted(f"{_format_bytes(size)} · {kind}")
            with If(STATE.ready):
                with If(STATE.kind == "image"):
                    Image(src=STATE.src, alt="Rendered media", width="100%")
                with If(STATE.kind == "audio"):
                    Audio(src=STATE.src)
                with If(STATE.kind == "video"):
                    Video(src=STATE.src, width="100%")
                Link("Download", href=STATE.src)
            with Else():
                with If(STATE.error):
                    Muted(STATE.error)
                with Else():
                    Muted("Opening the player…")
            Small(path)
    mount = None
    if playable:
        mount = CallTool(
            "load_playback",
            arguments={"path": path},
            on_success=[
                SetState("src", RESULT.src),
                SetState("ready", RESULT.ready),
                SetState("kind", RESULT.kind),
                SetState("error", RESULT.error),
            ],
            on_error=ShowToast(ERROR, variant="error"),
        )
    return PrefabApp(
        view=view,
        state={
            "src": "",
            "ready": False,
            "kind": kind,
            "error": "" if playable else playback_payload(path)["error"],
        },
        on_mount=mount,
    )


@MEDIA_UPLOAD.tool()
def load_playback(path: str) -> dict:
    """Read a rendered file for the download card. The model does not call this."""
    return playback_payload(path)


@MEDIA_UPLOAD.ui(
    description=(
        "Render a clip and open a player with a download button in the chat. "
        "Use this when the user wants the finished file in the conversation. "
        "Use export_clip when they want a specific path. Does not return file bytes."
    ),
    tags={"core"},
    annotations=EDIT,
)
async def download_media(clip_id: str, format: str, ctx: Context,
                         fps: Optional[float] = None,
                         codec: Optional[str] = None,
                         bitrate: Optional[str] = None,
                         time_seconds: float = 0.0,
                         at_end: bool = False,
                         preset: Optional[str] = None) -> ToolResult:
    """Render a clip into the chat as a player and download button.

    Args:
        clip_id: The clip to render.
        format: Output extension without the dot (mp4, mp3, gif, png).
            A still format renders one frame.
        fps: Frames per second for video. Ignored for stills.
        codec: Optional codec override. Ignored for stills.
        bitrate: Optional bitrate. Ignored for stills.
        time_seconds: Frame time, for stills only.
        at_end: Grab the last frame instead of time_seconds.
        preset: Delivery preset, same names as export_clip. Ignored for stills.
    """
    entry = _get(clip_id)
    ext = "." + format.strip().lstrip(".").lower()
    if not format.strip():
        raise ValueError("format is required (e.g. 'mp4', 'mp3', 'png').")
    is_still = ext in _STILL_IMAGE_EXTS
    if is_still and entry.kind not in ("video", "image"):
        raise ValueError(
            f"format '{format}' needs a video or image clip_id, but "
            f"'{clip_id}' is a {entry.kind} clip.")

    out_path = os.path.join(
        workspace_dir("exports"),
        f"{clip_id}_{uuid.uuid4().hex[:8]}{ext}",
    )
    loop = asyncio.get_running_loop()
    if is_still:
        t = _last_frame_time(entry.clip) if at_end else time_seconds

        def grab() -> None:
            with LOCK:
                entry.clip.save_frame(out_path, t=t)

        await loop.run_in_executor(None, grab)
    else:
        await ctx.info(f"Rendering {clip_id} as {ext}")
        await ctx.report_progress(0, None, "starting render")
        report = _bridge_progress(ctx)
        cancel = threading.Event()
        logger = _moviepy_logger(report, cancel)

        def work() -> None:
            with LOCK:
                _write_clip(entry, out_path, fps, codec, bitrate, logger, preset)

        try:
            await loop.run_in_executor(None, work)
        except asyncio.CancelledError:
            cancel.set()
            raise
        await ctx.report_progress(1, 1, "render complete")

    size = os.path.getsize(out_path)
    kind = playback_kind(out_path)
    card = _download_card(out_path, size, kind)
    if size <= PLAYBACK_LIMIT:
        text = (
            f"Opened a player for {os.path.basename(out_path)} "
            f"({_format_bytes(size)}). Saved at {out_path}."
        )
    else:
        text = (
            f"Saved {os.path.basename(out_path)} ({_format_bytes(size)}) "
            f"at {out_path}. Too large to play in the chat."
        )
    return ToolResult(
        content=[TextContent(type="text", text=text)],
        structured_content=_prefab_to_json(card, fastmcp_app_name=MEDIA_UPLOAD.name),
    )
