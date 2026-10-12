"""Loading, creating, and inspecting clips."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from typing import Any, Optional

from datetime import datetime, timezone
from moviepy_mcp.config import get_settings
from fastmcp import Context
from fastmcp.apps.file_upload import FileUpload
from fastmcp.tools.tool import ToolResult
from fastmcp.utilities.types import Image as MCPImage
from prefab_ui.actions import SetState, ShowToast
from prefab_ui.actions.mcp import CallTool
from prefab_ui.app import PrefabApp
from prefab_ui.components import (
    H3,
    Badge,
    Button,
    Card,
    CardContent,
    CardFooter,
    CardHeader,
    Column,
    DropZone,
    Muted,
    Row,
    Separator,
    Small,
    Text,
)
from prefab_ui.components.control_flow import Else, ForEach, If
from prefab_ui.rx import ERROR, RESULT, STATE, Rx
from moviepy import AudioFileClip, ColorClip, CompositeVideoClip, ImageClip, VideoFileClip, afx, concatenate_videoclips, vfx
from mcp.types import ToolAnnotations

from moviepy_mcp.paths import font_dirs, resolve_dir, resolve_read, workspace_dir
from moviepy_mcp.registry import _describe, _get, _register, current_ids, project_report, remove
from moviepy_mcp.runtime import LOCK, serialized
from moviepy_mcp.server import mcp
from moviepy_mcp.editing import _bridge_progress, _decode_base64, _detect_kind, _download_with_ytdlp, _make_text_clip, _preview_png, _validate_download_url

READ = ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False)
EDIT = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False)
DESTROY = ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=False)
NET = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=True)


def _store_upload(filename: str, raw: bytes, kind: Optional[str] = None,
                  label: Optional[str] = None,
                  output_dir: Optional[str] = None) -> tuple[str, str]:
    """Write uploaded bytes to disk and load them into the clip registry.

    Returns (clip_id, saved_path).
    """
    filename = os.path.basename((filename or "").strip())
    if not filename:
        raise ValueError("filename is required (e.g. 'clip.mp4').")
    ext = os.path.splitext(filename)[1].lower()
    detected = kind or _detect_kind(ext)
    if detected not in ("video", "audio", "image"):
        raise ValueError(
            f"Cannot determine media kind from filename '{filename}'. "
            "Pass kind='video', 'audio', or 'image' explicitly.")

    dest_dir = resolve_dir(output_dir) if output_dir else workspace_dir("uploads")
    dest_path = os.path.join(dest_dir, f"{uuid.uuid4().hex[:8]}_{filename}")
    with open(dest_path, "wb") as fh:
        fh.write(raw)

    if detected == "video":
        clip: Any = VideoFileClip(dest_path)
    elif detected == "audio":
        clip = AudioFileClip(dest_path)
    else:
        clip = ImageClip(dest_path)
    clip_id = _register(
        clip, detected, label or filename, None, f"upload_media({filename})",
        source_path=dest_path, args={"path": dest_path},
    )
    return clip_id, dest_path

def _format_upload_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes / (1024 * 1024):.1f} MB"

@mcp.tool(tags={'core'}, annotations=EDIT)
@serialized
def load_video(path: str, label: Optional[str] = None) -> dict:
    """Load a video file (mp4, mov, avi, webm, mkv...) into the registry.

    Args:
        path: Absolute path to the video file.
        label: Optional human-readable name for the clip.

    Returns:
        Clip metadata including the new ``clip_id`` to use in later calls.
    """
    path = resolve_read(path)
    clip = VideoFileClip(path)
    clip_id = _register(
        clip, "video", label or os.path.basename(path), None,
        f"load_video({path})", source_path=path, args={"path": path},
    )
    return _describe(clip_id)

@mcp.tool(tags={'core'}, annotations=NET)
@serialized
async def download_video(url: str, ctx: Context,
                         output_dir: Optional[str] = None,
                         label: Optional[str] = None,
                         cookies_from_browser: Optional[str] = None,
                         cookies_file: Optional[str] = None) -> dict:
    """Download a YouTube or Instagram video and load it into the registry.

    Uses yt-dlp. Public videos work without cookies; private/login-gated
    Instagram posts may need ``cookies_from_browser`` (e.g. ``chrome``) or
    ``cookies_file``.

    Args:
        url: Full YouTube or Instagram video/reel URL.
        output_dir: Where to save the file (created if missing). Defaults to
            a ``moviepy-mcp-downloads`` folder under the system temp dir.
        label: Optional human-readable name for the loaded clip.
        cookies_from_browser: Browser name for cookie import (chrome, firefox,
            edge, brave, etc.) — useful for Instagram.
        cookies_file: Path to a Netscape-format cookies.txt instead.

    Returns:
        Clip metadata plus ``downloaded_path``, ``source_url``, and ``title``.
    """
    url = _validate_download_url(url)
    if cookies_from_browser and cookies_file:
        raise ValueError(
            "Pass only one of cookies_from_browser or cookies_file, not both.")
    if cookies_from_browser and not get_settings().allow_browser_cookies:
        raise ValueError(
            "Browser cookie import is off. Set MOVIEPY_MCP_ALLOW_BROWSER_COOKIES=1 "
            "to enable it for this process."
        )
    if cookies_file:
        cookies_file = resolve_read(cookies_file)
    dest = resolve_dir(output_dir) if output_dir else workspace_dir("downloads")
    await ctx.info(f"Downloading {url}")
    report = _bridge_progress(ctx)
    loop = asyncio.get_running_loop()

    def work() -> tuple[str, dict[str, Any]]:
        with LOCK:
            return _download_with_ytdlp(
                url, dest,
                cookies_from_browser=cookies_from_browser,
                cookies_file=cookies_file,
                on_progress=report,
            )

    path, info = await loop.run_in_executor(None, work)
    title = info.get("title") or os.path.basename(path)
    await ctx.info("Loading downloaded file into registry")

    def open_clip() -> Any:
        with LOCK:
            return VideoFileClip(path)

    clip = await loop.run_in_executor(None, open_clip)
    clip_id = _register(
        clip, "video", label or title, None, f"download_video({url})",
        source_path=path, args={"path": path},
    )
    meta = _describe(clip_id)
    meta["downloaded_path"] = path
    meta["source_url"] = url
    meta["title"] = title
    return meta

@mcp.tool(tags={'core'}, annotations=EDIT)
@serialized
def upload_media(filename: str, data_base64: str,
                 kind: Optional[str] = None,
                 output_dir: Optional[str] = None,
                 label: Optional[str] = None) -> dict:
    """Upload raw media bytes and load them into the registry.

    Use this instead of load_video/load_audio/load_image when the calling
    client has the file's bytes in memory but no filesystem path the server
    can read (e.g. a browser-based or remote MCP client). Encode the file
    as base64 and pass it here; the bytes are written to a server-managed
    file (moviepy needs a real file on disk to stream from) and loaded.

    Args:
        filename: Original filename, used to pick the file extension and
            (unless kind is given) to guess whether it's video/audio/image.
        data_base64: The file contents, base64-encoded.
        kind: Force 'video', 'audio', or 'image' instead of guessing from
            filename's extension.
        output_dir: Where to save the decoded file. Defaults to a
            moviepy-mcp-uploads folder under the system temp dir.
        label: Optional human-readable name for the clip.

    Returns:
        Clip metadata plus ``saved_path`` and ``file_size_bytes``.
    """
    raw = _decode_base64(data_base64)
    clip_id, dest_path = _store_upload(filename, raw, kind=kind, label=label,
                                       output_dir=output_dir)
    meta = _describe(clip_id)
    meta["saved_path"] = dest_path
    meta["file_size_bytes"] = len(raw)
    return meta

@mcp.tool(tags={'core'}, annotations=EDIT)
@serialized
def load_audio(path: str, label: Optional[str] = None) -> dict:
    """Load an audio file (mp3, wav, aac, ogg...) into the registry.

    Args:
        path: Absolute path to the audio file.
        label: Optional human-readable name for the clip.

    Returns:
        Clip metadata including the new ``clip_id``.
    """
    path = resolve_read(path)
    clip = AudioFileClip(path)
    clip_id = _register(
        clip, "audio", label or os.path.basename(path), None,
        f"load_audio({path})", source_path=path, args={"path": path},
    )
    return _describe(clip_id)

@mcp.tool(tags={'core'}, annotations=EDIT)
@serialized
def load_image(path: str, duration_seconds: float = 5.0,
               label: Optional[str] = None) -> dict:
    """Load a still image (png, jpg...) as a video clip of fixed duration.

    Useful for slideshows, intros, watermark sources, and overlays.

    Args:
        path: Absolute path to the image file.
        duration_seconds: How long the image should display when used as video.
        label: Optional human-readable name.

    Returns:
        Clip metadata including the new ``clip_id``.
    """
    path = resolve_read(path)
    clip = ImageClip(path).with_duration(duration_seconds)
    clip_id = _register(
        clip, "image", label or os.path.basename(path), None,
        f"load_image({path}, {duration_seconds}s)",
        source_path=path,
        args={"path": path, "duration_seconds": duration_seconds},
    )
    return _describe(clip_id)

@mcp.tool(tags={'core'}, annotations=EDIT)
@serialized
def create_color_clip(width: int, height: int, color_rgb: list[int],
                      duration_seconds: float,
                      label: Optional[str] = None) -> dict:
    """Create a solid-color video clip (backgrounds, spacers, title cards).

    Args:
        width: Width in pixels.
        height: Height in pixels.
        color_rgb: Color as [R, G, B], each 0-255 (e.g. [0, 0, 0] for black).
        duration_seconds: Clip duration.
        label: Optional human-readable name.
    """
    clip = ColorClip(size=(width, height), color=tuple(color_rgb),
                     duration=duration_seconds)
    clip_id = _register(clip, "video", label or "color_clip", None,
                        f"create_color_clip({width}x{height})")
    return _describe(clip_id)

@mcp.tool(tags={'core'}, annotations=EDIT)
@serialized
def create_text_clip(text: str, font_size: int = 48,
                     color: str = "white",
                     bg_color: Optional[str] = None,
                     font: Optional[str] = None,
                     duration_seconds: float = 5.0,
                     stroke_color: Optional[str] = None,
                     stroke_width: float = 0,
                     text_align: str = "left",
                     width: Optional[int] = None,
                     height: Optional[int] = None,
                     method: str = "label",
                     label: Optional[str] = None) -> dict:
    """Create a standalone text clip (titles, captions, credits).

    To place text ON TOP of a video, create it here and then use
    ``overlay_clip`` with position and timing.

    Args:
        text: The text to render. Use \\n for line breaks.
        font_size: Font size in points.
        color: Text color name or hex (e.g. 'white', '#FFD700').
        bg_color: Optional background color; transparent if omitted.
        font: Optional path to a .ttf/.otf font file. Uses a default font
            if omitted.
        duration_seconds: How long the text displays.
        stroke_color: Optional outline color (e.g. 'black').
        stroke_width: Outline thickness in pixels.
        text_align: 'left', 'center', or 'right' (used when wrapping).
        width: Wrap width in pixels. Sets method to caption when given.
        height: Optional box height for caption layout.
        method: 'label' (one box) or 'caption' (word-wrap to width).
        label: Optional human-readable name.
    """
    clip = _make_text_clip(
        text=text, font_size=font_size, color=color, bg_color=bg_color,
        font=font, duration_seconds=duration_seconds,
        stroke_color=stroke_color, stroke_width=stroke_width,
        text_align=text_align, width=width, height=height, method=method,
    )
    clip_id = _register(clip, "video", label or f"text:{text[:24]}", None,
                        "create_text_clip")
    return _describe(clip_id)

@mcp.tool(tags={'core'}, annotations=EDIT)
@serialized
def create_slideshow(image_paths: list[str], duration_seconds: float = 3.0,
                     transition_seconds: float = 0.0,
                     audio_clip_id: Optional[str] = None,
                     label: Optional[str] = None) -> dict:
    """Build a video slideshow from a list of images in one call.

    Loads each image, shows it for duration_seconds, optionally crossfades
    between slides, and optionally lays background audio under the whole
    thing (looped or trimmed to match the slideshow length).

    Args:
        image_paths: Absolute paths to image files, in slide order (>= 2).
        duration_seconds: How long each image displays (before any overlap).
        transition_seconds: Crossfade overlap between consecutive slides;
            0 = hard cuts.
        audio_clip_id: Optional audio clip_id (from load_audio) to attach as
            background music, looped or trimmed to the slideshow's length.
        label: Optional human-readable name.
    """
    if len(image_paths) < 2:
        raise ValueError("Provide at least two image_paths.")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be > 0.")
    if transition_seconds < 0:
        raise ValueError("transition_seconds must be >= 0.")
    if transition_seconds >= duration_seconds:
        raise ValueError(
            "transition_seconds must be less than duration_seconds.")

    clips = [ImageClip(resolve_read(p)).with_duration(duration_seconds)
             for p in image_paths]

    if transition_seconds > 0:
        placed = []
        t = 0.0
        last = len(clips) - 1
        for i, clip in enumerate(clips):
            effects = []
            if i > 0:
                effects.append(vfx.CrossFadeIn(transition_seconds))
            if i < last:
                effects.append(vfx.CrossFadeOut(transition_seconds))
            if effects:
                clip = clip.with_effects(effects)
            placed.append(clip.with_start(t))
            t += duration_seconds - transition_seconds
        video = CompositeVideoClip(placed)
    else:
        video = concatenate_videoclips(clips, method="compose")

    parents = None
    if audio_clip_id is not None:
        aentry = _get(audio_clip_id, expect=("audio",))
        audio = aentry.clip
        if audio.duration and audio.duration < video.duration:
            audio = audio.with_effects([afx.AudioLoop(duration=video.duration)])
        elif audio.duration and audio.duration > video.duration:
            audio = audio.subclipped(0, video.duration)
        video = video.with_audio(audio)
        parents = [aentry]

    nid = _register(video, "video", label or "slideshow", None,
                    f"create_slideshow({len(image_paths)} images)",
                    parents=parents)
    return _describe(nid)

@mcp.tool(tags={'core'}, annotations=READ)
@serialized
def list_clips() -> dict:
    """List every clip currently in the registry with its metadata."""
    ids = current_ids()
    return {"count": len(ids), "clips": [_describe(cid) for cid in ids]}

@mcp.tool(tags={'core'}, annotations=READ)
@serialized
def get_clip_info(clip_id: str) -> dict:
    """Get full metadata for one clip: duration, size, fps, audio, history."""
    _get(clip_id)
    return _describe(clip_id)

@mcp.tool(tags={'core'}, annotations=READ)
@serialized
def preview_frame(clip_id: str, time_seconds: float = 0.0,
                  max_width: int = 640) -> ToolResult:
    """Return a still frame as an image so the model can see the edit.

    Does not write a file. Downscales so the image fits in context; pass
    max_width=0 for native resolution.

    Args:
        clip_id: Video or image clip.
        time_seconds: Timestamp to grab (clamped to the clip duration).
        max_width: Max preview width in pixels. 0 = do not downscale.
            Default 640.
    """
    if max_width < 0:
        raise ValueError("max_width must be >= 0 (0 = native resolution).")
    entry = _get(clip_id, expect=("video", "image"))
    duration = entry.clip.duration
    if time_seconds < 0:
        raise ValueError(f"time_seconds must be >= 0, got {time_seconds}.")
    if duration is not None and time_seconds > duration:
        raise ValueError(
            f"time_seconds {time_seconds} is past duration {duration}s. "
            f"Use a time in [0, {duration}].")
    data, w, h = _preview_png(entry.clip, time_seconds, max_width)
    meta = _describe(clip_id)
    meta["time_seconds"] = time_seconds
    meta["preview_size"] = {"width": w, "height": h}
    return ToolResult(
        content=[MCPImage(data=data, format="png").to_image_content()],
        structured_content=meta,
    )

@mcp.tool(tags={'core'}, annotations=DESTROY)
@serialized
def delete_clip(clip_id: str) -> dict:
    """Remove a clip from the registry and free its resources."""
    remaining = remove(clip_id)
    return {"deleted": clip_id, "remaining_clips": remaining}

@mcp.resource("clip://{clip_id}/info", mime_type="application/json", tags={'core'}, annotations={'readOnlyHint': True, 'idempotentHint': True})
@serialized
def clip_info_resource(clip_id: str) -> str:
    """JSON metadata for a registered clip."""
    _get(clip_id)
    return json.dumps(_describe(clip_id))

@mcp.resource("clip://{clip_id}/frame", mime_type="image/png", tags={'core'}, annotations={'readOnlyHint': True, 'idempotentHint': True})
@serialized
def clip_frame_resource(clip_id: str) -> bytes:
    """PNG of the first frame (downscaled to 640px). Use preview_frame for other times."""
    entry = _get(clip_id, expect=("video", "image"))
    data, _, _ = _preview_png(entry.clip, 0.0, max_width=640)
    return data

@mcp.tool(tags={"core"}, annotations=READ)
@serialized
def list_fonts(limit: int = 200) -> dict:
    """List font files create_text_clip can use. Returns name and absolute path."""
    if limit < 1:
        raise ValueError("limit must be >= 1.")
    fonts: list[dict[str, str]] = []
    suffixes = {".ttf", ".otf", ".ttc"}
    for folder in font_dirs():
        for path in folder.rglob("*"):
            if path.suffix.lower() not in suffixes:
                continue
            fonts.append({"name": path.stem, "path": str(path)})
            if len(fonts) >= limit:
                return {"count": len(fonts), "fonts": fonts}
    return {"count": len(fonts), "fonts": fonts}


@mcp.tool(tags={"core"}, annotations=READ)
@serialized
def project_status() -> dict:
    """Report source clips restored from the project file, and edits that were not rebuilt.

    Restarting the server reopens load/upload/download files. Derived edits stay
    in the project log until you run them again.
    """
    return project_report()


class MediaUpload(FileUpload):
    """Drag-and-drop video/audio/image upload rendered inside the chat UI.

    Built on fastmcp's ``FileUpload`` app (requires ``fastmcp[apps]``).
    Dropped files never enter the model's context window: bytes are
    decoded straight to a server-side file and loaded into the clip
    registry via ``_store_upload`` (the same path ``upload_media`` uses),
    and only the resulting ``clip_id`` plus lightweight metadata are
    surfaced to the model.
    """

    def __init__(self) -> None:
        super().__init__(
            name="Media",
            max_file_size=500 * 1024 * 1024,
            title="Upload Media",
            description=(
                "Drop a video, audio, or image file. It loads as a clip "
                "and the clip_id shows on the row below."
            ),
            drop_label="Drop video, audio, or image files here",
        )
        # The stock card hides clip_id. Swap in one that shows it.
        self._local._components.pop("tool:file_manager@", None)
        self._local._components.pop("tool:list_files@", None)
        self._install_upload_ui()

    def _install_upload_ui(self) -> None:
        provider = self

        @self.tool(model=True)
        def list_files(ctx: Context) -> list[dict]:
            """List files from the upload card. Each item includes clip_id."""
            return provider.on_list(ctx)

        @self.ui(
            description=(
                "Open a drag-and-drop card so the user can upload video, "
                "audio, or image files. Each file becomes a clip. After "
                "they upload, call list_files and use the clip_id. Do not "
                "ask the user to paste file bytes."
            ),
            tags={"core"},
            annotations=READ,
        )
        def file_manager(ctx: Context) -> PrefabApp:
            """Open the upload card."""
            with Card(css_class="max-w-2xl mx-auto") as view:
                with CardHeader(), Row(gap=2, align="center"):
                    H3(provider._title)
                    with If(STATE.stored.length()):
                        Badge(STATE.stored.length(), variant="secondary")

                with CardContent(), Column(gap=4):
                    Muted(provider._description)
                    DropZone(
                        name="pending",
                        icon="inbox",
                        label=provider._drop_label,
                        description=(
                            "Video, audio, or image, up to "
                            f"{_format_upload_size(provider._max_file_size)}"
                        ),
                        accept="video/*,audio/*,image/*",
                        multiple=True,
                        max_size=provider._max_file_size,
                    )
                    with If(STATE.pending.length()), Column(gap=2):
                        with (
                            ForEach("pending"),
                            Row(gap=2, align="center"),
                            Column(gap=0),
                        ):
                            Small(Rx("$item.name"))
                            Muted(Rx("$item.type"))
                        Button(
                            "Upload",
                            on_click=CallTool(
                                "store_files",
                                arguments={"files": Rx("pending")},
                                on_success=[
                                    SetState("stored", RESULT),
                                    SetState("pending", []),
                                    ShowToast("Files uploaded", variant="success"),
                                ],
                                on_error=ShowToast(ERROR, variant="error"),
                            ),
                        )
                    with If(STATE.stored.length()):
                        Separator()
                        Text("Uploaded", css_class="font-medium text-sm")
                        with (
                            ForEach("stored") as item,
                            Row(gap=2, align="center", css_class="justify-between"),
                        ):
                            with Column(gap=0):
                                Small(item.name)
                                Muted(item.clip_id)
                            with Row(gap=2):
                                Badge(item.type, variant="secondary")
                                Badge(item.size_display, variant="outline")

                with CardFooter(), Row(align="center", css_class="w-full"):
                    with If(STATE.stored.length()):
                        Muted(
                            f"{STATE.stored.length()}"
                            f" {STATE.stored.length().pluralize('file')}"
                            " ready"
                        )
                    with Else():
                        Muted("No files uploaded yet")
            return PrefabApp(
                view=view,
                state={"pending": [], "stored": provider.on_list(ctx)},
            )

    def _summarize(self, entry: dict[str, Any]) -> dict[str, Any]:
        return {
            "name": entry["name"],
            "type": entry["type"],
            "file_size_bytes": entry["file_size_bytes"],
            "size_display": _format_upload_size(entry["file_size_bytes"]),
            "uploaded_at": entry["uploaded_at"],
            "clip_id": entry["clip_id"],
        }

    def on_store(self, files: list[dict[str, Any]],
                ctx: Context) -> list[dict[str, Any]]:
        scope = self._get_scope_key(ctx)
        session_files = self._store.setdefault(scope, {})
        for f in files:
            name = os.path.basename(f.get("name") or "")
            raw = _decode_base64(f.get("data", ""))
            clip_id, _saved_path = _store_upload(name, raw)
            session_files[name] = {
                "name": name,
                "type": f.get("type") or _get(clip_id).kind,
                "file_size_bytes": len(raw),
                "clip_id": clip_id,
                "uploaded_at": datetime.now(timezone.utc)
                .isoformat(timespec="seconds"),
            }
        return [self._summarize(e) for e in session_files.values()]

    def on_list(self, ctx: Context) -> list[dict[str, Any]]:
        scope = self._get_scope_key(ctx)
        return [self._summarize(e) for e in self._store.get(scope, {}).values()]

    def on_read(self, name: str, ctx: Context) -> dict[str, Any]:
        scope = self._get_scope_key(ctx)
        session_files = self._store.get(scope, {})
        if name not in session_files:
            available = list(session_files)
            raise ValueError(f"File {name!r} not found. Available: {available}")
        entry = session_files[name]
        result = self._summarize(entry)
        result.update(_describe(entry["clip_id"]))
        return result

MEDIA_UPLOAD = MediaUpload()
