"""MoviePy MCP Server.

An LLM-friendly Model Context Protocol server for video and audio editing,
built on MoviePy v2 and FastMCP.

Design principles
-----------------
1. In-memory clip registry: every operation returns a new ``clip_id``.
   Operations are chained by passing clip_ids; nothing is written to disk
   until ``export_clip`` is called. This keeps multi-step edits fast.
2. Non-destructive: source clips are never mutated. Each edit produces a
   new registry entry, so an agent can branch or retry freely.
3. Structured, self-describing results: editing tools return a dict with
   ``clip_id`` plus metadata (duration, size, fps, has_audio) so the
   calling model always knows the current state without extra calls.
   ``preview_frame`` also returns an image so the model can see the frame.
4. Fail loudly with actionable messages: errors name the offending
   argument and suggest the fix.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import re
import tempfile
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urlparse

from fastmcp import Context, FastMCP
from fastmcp.tools.tool import ToolResult
from fastmcp.utilities.types import Image as MCPImage
from moviepy import (
    AudioFileClip,
    ColorClip,
    CompositeAudioClip,
    CompositeVideoClip,
    ImageClip,
    TextClip,
    VideoFileClip,
    afx,
    clips_array,
    concatenate_audioclips,
    concatenate_videoclips,
    vfx,
)

_POSITIONS = {
    "center": "center", "top": ("center", "top"),
    "bottom": ("center", "bottom"), "left": ("left", "center"),
    "right": ("right", "center"), "top-left": ("left", "top"),
    "top-right": ("right", "top"), "bottom-left": ("left", "bottom"),
    "bottom-right": ("right", "bottom"),
}
_SLIDE_SIDES = ("top", "bottom", "left", "right")
_TS_RE = re.compile(r"(\d{1,2}):(\d{2}):(\d{2})[.,](\d{1,3})")

mcp = FastMCP(
    name="moviepy",
    instructions=(
        "Video/audio editing server. Workflow: load media with load_video / "
        "load_audio / load_image, or fetch from YouTube/Instagram with "
        "download_video (returns a clip_id), transform it with the editing "
        "tools (each returns a NEW clip_id — always use the latest id for the "
        "next step), call preview_frame to see a still, then write the result "
        "with export_clip. Use list_clips / get_clip_info / clip://{id}/info "
        "to inspect state at any time."
    ),
)

# --------------------------------------------------------------------------- #
# Clip registry
# --------------------------------------------------------------------------- #


@dataclass
class ClipEntry:
    clip: Any
    kind: str  # "video" | "audio" | "image"
    label: str
    history: list[str] = field(default_factory=list)


_REGISTRY: dict[str, ClipEntry] = {}


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _get(clip_id: str, expect: Optional[tuple[str, ...]] = None) -> ClipEntry:
    entry = _REGISTRY.get(clip_id)
    if entry is None:
        known = ", ".join(_REGISTRY) or "(registry is empty)"
        raise ValueError(
            f"Unknown clip_id '{clip_id}'. Known clip_ids: {known}. "
            "Load media first with load_video/load_audio/load_image."
        )
    if expect and entry.kind not in expect:
        raise ValueError(
            f"clip_id '{clip_id}' is a {entry.kind} clip but this tool "
            f"requires one of: {', '.join(expect)}."
        )
    return entry


def _register(clip: Any, kind: str, label: str, parent: Optional[ClipEntry],
              op: str) -> str:
    clip_id = _new_id(kind)
    history = (parent.history[:] if parent else []) + [op]
    _REGISTRY[clip_id] = ClipEntry(clip=clip, kind=kind, label=label,
                                   history=history)
    return clip_id


def _describe(clip_id: str) -> dict[str, Any]:
    """Uniform metadata payload returned by every tool."""
    entry = _REGISTRY[clip_id]
    clip = entry.clip
    info: dict[str, Any] = {
        "clip_id": clip_id,
        "kind": entry.kind,
        "label": entry.label,
        "duration_seconds": round(clip.duration, 3) if clip.duration else None,
        "history": entry.history,
    }
    if entry.kind in ("video", "image"):
        info["size"] = {"width": int(clip.w), "height": int(clip.h)}
        fps = getattr(clip, "fps", None)
        info["fps"] = float(fps) if fps is not None else None
        info["has_audio"] = clip.audio is not None
    return info


def _check_path(path: str) -> str:
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        raise ValueError(f"File not found: '{path}'. Provide an absolute path "
                         "to an existing file.")
    return path


_YT_HOSTS = ("youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be",
             "www.youtu.be", "music.youtube.com")
_IG_HOSTS = ("instagram.com", "www.instagram.com")


def _validate_download_url(url: str) -> str:
    """Accept YouTube or Instagram http(s) URLs; raise with a clear fix otherwise."""
    url = (url or "").strip()
    if not url:
        raise ValueError("url is required (YouTube or Instagram video URL).")
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(
            f"Invalid url '{url}'. Provide a full http(s) YouTube or Instagram URL.")
    host = parsed.netloc.lower().split("@")[-1]
    if host.startswith("www."):
        host = host[4:]
    allowed = {h.removeprefix("www.") for h in (_YT_HOSTS + _IG_HOSTS)}
    if host not in allowed:
        raise ValueError(
            f"Unsupported host '{parsed.netloc}'. download_video supports YouTube "
            "and Instagram only (e.g. https://www.youtube.com/watch?v=... or "
            "https://www.instagram.com/reel/...).")
    return url


def _resolve_downloaded_path(info: dict[str, Any], prepared: str) -> str:
    """Find the on-disk file after yt-dlp download (handles merge/ext changes)."""
    candidates: list[str] = []
    for entry in info.get("requested_downloads") or []:
        fp = entry.get("filepath")
        if fp:
            candidates.append(fp)
    if info.get("filepath"):
        candidates.append(info["filepath"])
    if info.get("_filename"):
        candidates.append(info["_filename"])
    candidates.append(prepared)
    root, _ = os.path.splitext(prepared)
    for ext in (".mp4", ".mkv", ".webm", ".mov", ".m4a", ".mp3"):
        candidates.append(root + ext)

    seen: set[str] = set()
    for path in candidates:
        if not path or path in seen:
            continue
        seen.add(path)
        if os.path.isfile(path):
            return path
    raise ValueError(
        f"Download finished but no output file was found near '{prepared}'.")


def _download_with_ytdlp(url: str, output_dir: str,
                         cookies_from_browser: Optional[str] = None,
                         cookies_file: Optional[str] = None,
                         on_progress: Optional[Callable[
                             [float, Optional[float], str], None]] = None,
                         ) -> tuple[str, dict]:
    """Download a single video; return (filepath, info_dict)."""
    import yt_dlp

    os.makedirs(output_dir, exist_ok=True)
    # Sanitize template: avoid path-breaking chars in titles
    outtmpl = os.path.join(output_dir, "%(title).80B [%(id)s].%(ext)s")
    opts: dict[str, Any] = {
        "outtmpl": outtmpl,
        "format": "bv*+ba/b",
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }
    if on_progress is not None:
        opts["progress_hooks"] = [_ytdlp_progress_hook(on_progress)]
    if cookies_file:
        path = os.path.expanduser(cookies_file)
        if not os.path.isfile(path):
            raise ValueError(f"cookies_file not found: '{path}'.")
        opts["cookiefile"] = path
    if cookies_from_browser:
        # yt-dlp expects a tuple like ("chrome",) or ("chrome", "Profile 1", ...)
        opts["cookiesfrombrowser"] = (cookies_from_browser,)

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if info is None:
                raise ValueError("yt-dlp returned no info for this URL.")
            if "entries" in info:
                entries = [e for e in (info.get("entries") or []) if e]
                if not entries:
                    raise ValueError("URL looks like a playlist/feed with no entries.")
                info = entries[0]
            prepared = ydl.prepare_filename(info)
            path = _resolve_downloaded_path(info, prepared)
            return path, info
    except yt_dlp.utils.DownloadError as exc:
        raise ValueError(
            f"Download failed for '{url}': {exc}. "
            "For private Instagram posts, pass cookies_from_browser "
            "(e.g. 'chrome') or cookies_file."
        ) from exc


def _pil_filter(clip: Any, filter_fn: Any) -> Any:
    """Apply a Pillow filter function to every frame of a clip."""
    import numpy as np
    from PIL import Image

    def fl(im: Any) -> Any:
        arr = np.asarray(im)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        out = filter_fn(Image.fromarray(arr))
        return np.asarray(out)

    return clip.image_transform(fl)


def _ensure_uint8(clip: Any) -> Any:
    """Cast frames to uint8 so Pillow-backed MoviePy effects work on ColorClips."""
    import numpy as np

    def fl(im: Any) -> Any:
        arr = np.asarray(im)
        if arr.dtype == np.uint8:
            return arr
        return np.clip(arr, 0, 255).astype(np.uint8)

    return clip.image_transform(fl)


def _preview_png(clip: Any, time_seconds: float,
                 max_width: int) -> tuple[bytes, int, int]:
    """Render one frame to PNG bytes, optionally downscaled for the model."""
    import numpy as np
    from PIL import Image as PILImage

    if time_seconds < 0:
        raise ValueError(f"time_seconds must be >= 0, got {time_seconds}.")
    duration = getattr(clip, "duration", None)
    t = time_seconds
    if duration is not None and duration > 0:
        # get_frame at t == duration is out of range
        t = min(t, max(0.0, duration - 1e-6))
    frame = clip.get_frame(t)
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = PILImage.fromarray(arr)
    w, h = img.size
    if max_width > 0 and w > max_width:
        h = max(1, round(h * max_width / w))
        w = max_width
        img = img.resize((w, h), PILImage.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), w, h


def _bridge_progress(
    ctx: Context,
) -> Callable[[float, Optional[float], str], None]:
    """Sync callback that schedules ctx.report_progress on the running loop."""
    loop = asyncio.get_running_loop()
    last_pct = {"v": -1}

    def report(progress: float, total: Optional[float], message: str) -> None:
        if total and total > 0:
            pct = int(100 * progress / total)
            if pct == last_pct["v"] and progress < total:
                return
            last_pct["v"] = pct
        try:
            asyncio.run_coroutine_threadsafe(
                ctx.report_progress(progress, total, message),
                loop,
            )
        except RuntimeError:
            pass

    return report


def _ytdlp_progress_hook(
    on_progress: Callable[[float, Optional[float], str], None],
) -> Callable[[dict[str, Any]], None]:
    """Map yt-dlp progress_hooks dicts onto (progress, total, message)."""

    def hook(d: dict[str, Any]) -> None:
        status = d.get("status")
        if status == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            done = float(d.get("downloaded_bytes") or 0)
            on_progress(done, float(total) if total else None, "downloading")
        elif status == "finished":
            done = float(d.get("total_bytes") or d.get("downloaded_bytes") or 1)
            on_progress(done, done, "finished")

    return hook


def _moviepy_logger(
    report: Callable[[float, Optional[float], str], None],
) -> Any:
    from proglog import ProgressBarLogger

    class Logger(ProgressBarLogger):
        def bars_callback(self, bar, attr, value, old_value=None):
            if attr != "index":
                return
            total = (self.bars.get(bar) or {}).get("total")
            if not total:
                return
            report(float(value), float(total), str(bar))

    return Logger(min_time_interval=0.25)


def _write_clip(entry: ClipEntry, output_path: str, fps: Optional[float],
                codec: Optional[str], bitrate: Optional[str],
                logger: Any) -> None:
    ext = os.path.splitext(output_path)[1].lower()
    if entry.kind == "audio":
        kwargs: dict[str, Any] = {"logger": logger}
        if bitrate:
            kwargs["bitrate"] = bitrate
        entry.clip.write_audiofile(output_path, **kwargs)
        return
    if ext == ".gif":
        entry.clip.write_gif(output_path, fps=fps or 12, logger=logger)
        return
    clip = entry.clip
    out_fps = fps or getattr(clip, "fps", None) or 24
    kwargs = {"fps": out_fps, "logger": logger}
    if codec:
        kwargs["codec"] = codec
    if bitrate:
        kwargs["bitrate"] = bitrate
    clip.write_videofile(output_path, **kwargs)


def _with_position(clip: Any, position: str = "center",
                   x: Optional[int] = None, y: Optional[int] = None) -> Any:
    if x is not None and y is not None:
        return clip.with_position((x, y))
    if position not in _POSITIONS:
        raise ValueError(
            f"Unknown position '{position}'. Valid: {', '.join(_POSITIONS)}.")
    return clip.with_position(_POSITIONS[position])


def _parse_timestamp(value: str) -> float:
    match = _TS_RE.search(value.strip())
    if not match:
        raise ValueError(
            f"Cannot parse timestamp '{value}'. Expected H:MM:SS.mmm or "
            "H:MM:SS,mmm.")
    hours, minutes, seconds, frac = match.groups()
    frac = frac.ljust(3, "0")[:3]
    return (int(hours) * 3600 + int(minutes) * 60 + int(seconds)
            + int(frac) / 1000.0)


def _parse_subtitles(text: str) -> list[tuple[float, float, str]]:
    """Parse SRT or VTT into (start, end, text) cues."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        raise ValueError("Subtitle text is empty.")
    cues: list[tuple[float, float, str]] = []
    for block in re.split(r"\n\s*\n", text):
        lines = []
        for raw in block.split("\n"):
            line = raw.strip()
            if not line or line == "WEBVTT" or line.startswith("NOTE") \
                    or line.startswith("STYLE"):
                continue
            lines.append(line)
        idx = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if idx is None:
            continue
        start_raw, end_raw = lines[idx].split("-->", 1)
        end_raw = end_raw.strip().split()[0]
        start = _parse_timestamp(start_raw)
        end = _parse_timestamp(end_raw)
        body = re.sub(r"<[^>]+>", "", "\n".join(lines[idx + 1:])).strip()
        if body and end > start:
            cues.append((start, end, body))
    if not cues:
        raise ValueError(
            "No subtitle cues found. Provide SRT or WebVTT with --> timestamps.")
    return cues


def _make_text_clip(
    text: str,
    font_size: int = 48,
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
) -> Any:
    if width and method == "label":
        method = "caption"
    if method not in ("label", "caption"):
        raise ValueError("method must be 'label' (single line) or 'caption' (wrap).")
    if method == "caption" and not width:
        raise ValueError("caption method requires width (wrap width in pixels).")
    kwargs: dict[str, Any] = dict(
        text=text, font_size=font_size, color=color,
        stroke_width=stroke_width, method=method, text_align=text_align,
    )
    if bg_color:
        kwargs["bg_color"] = bg_color
    if font:
        kwargs["font"] = _check_path(font)
    if stroke_color:
        kwargs["stroke_color"] = stroke_color
    if width or height:
        kwargs["size"] = (width, height)
    try:
        return TextClip(**kwargs).with_duration(duration_seconds)
    except Exception as exc:
        raise ValueError(
            f"Text rendering failed: {exc}. Pass font= to a .ttf/.otf file."
        ) from exc


def _apply_audio_effect(entry: ClipEntry, effect: Any, op: str) -> dict:
    if entry.kind == "audio":
        new = entry.clip.with_effects([effect])
        kind = "audio"
    else:
        if entry.clip.audio is None:
            raise ValueError(
                f"clip_id has no audio track. Load audio or attach_audio first.")
        new = entry.clip.with_audio(entry.clip.audio.with_effects([effect]))
        kind = "video" if entry.kind == "image" else entry.kind
    nid = _register(new, kind, entry.label, entry, op)
    return _describe(nid)


def _ken_burns(clip: Any, start_zoom: float, end_zoom: float,
               start_x: float, start_y: float,
               end_x: float, end_y: float) -> Any:
    import numpy as np
    from PIL import Image as PILImage

    w, h = clip.w, clip.h
    duration = clip.duration or 1.0

    def fl(gf, t):
        p = 0.0 if duration <= 0 else min(1.0, max(0.0, t / duration))
        z = max(start_zoom + (end_zoom - start_zoom) * p, 1.0)
        cx = (start_x + (end_x - start_x) * p) * (w - 1)
        cy = (start_y + (end_y - start_y) * p) * (h - 1)
        cw, ch = w / z, h / z
        x1 = min(max(0.0, cx - cw / 2), max(0.0, w - cw))
        y1 = min(max(0.0, cy - ch / 2), max(0.0, h - ch))
        frame = np.asarray(gf(t))
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        img = PILImage.fromarray(frame)
        cropped = img.crop((
            int(round(x1)), int(round(y1)),
            int(round(x1 + cw)), int(round(y1 + ch)),
        ))
        if cropped.size != (w, h):
            cropped = cropped.resize((w, h), PILImage.Resampling.LANCZOS)
        return np.asarray(cropped)

    return clip.transform(fl)


# --------------------------------------------------------------------------- #
# Loading media
# --------------------------------------------------------------------------- #


@mcp.tool
def load_video(path: str, label: Optional[str] = None) -> dict:
    """Load a video file (mp4, mov, avi, webm, mkv...) into the registry.

    Args:
        path: Absolute path to the video file.
        label: Optional human-readable name for the clip.

    Returns:
        Clip metadata including the new ``clip_id`` to use in later calls.
    """
    path = _check_path(path)
    clip = VideoFileClip(path)
    clip_id = _register(clip, "video", label or os.path.basename(path), None,
                        f"load_video({path})")
    return _describe(clip_id)


@mcp.tool
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
    dest = os.path.expanduser(
        output_dir
        or os.path.join(tempfile.gettempdir(), "moviepy-mcp-downloads"))
    await ctx.info(f"Downloading {url}")
    report = _bridge_progress(ctx)
    loop = asyncio.get_running_loop()
    path, info = await loop.run_in_executor(
        None,
        lambda: _download_with_ytdlp(
            url, dest,
            cookies_from_browser=cookies_from_browser,
            cookies_file=cookies_file,
            on_progress=report,
        ),
    )
    title = info.get("title") or os.path.basename(path)
    await ctx.info("Loading downloaded file into registry")
    clip = VideoFileClip(path)
    clip_id = _register(
        clip, "video", label or title, None,
        f"download_video({url})")
    meta = _describe(clip_id)
    meta["downloaded_path"] = path
    meta["source_url"] = url
    meta["title"] = title
    return meta


@mcp.tool
def load_audio(path: str, label: Optional[str] = None) -> dict:
    """Load an audio file (mp3, wav, aac, ogg...) into the registry.

    Args:
        path: Absolute path to the audio file.
        label: Optional human-readable name for the clip.

    Returns:
        Clip metadata including the new ``clip_id``.
    """
    path = _check_path(path)
    clip = AudioFileClip(path)
    clip_id = _register(clip, "audio", label or os.path.basename(path), None,
                        f"load_audio({path})")
    return _describe(clip_id)


@mcp.tool
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
    path = _check_path(path)
    clip = ImageClip(path).with_duration(duration_seconds)
    clip_id = _register(clip, "image", label or os.path.basename(path), None,
                        f"load_image({path}, {duration_seconds}s)")
    return _describe(clip_id)


@mcp.tool
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


@mcp.tool
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


# --------------------------------------------------------------------------- #
# Inspection & registry management
# --------------------------------------------------------------------------- #


@mcp.tool
def list_clips() -> dict:
    """List every clip currently in the registry with its metadata."""
    return {"count": len(_REGISTRY),
            "clips": [_describe(cid) for cid in _REGISTRY]}


@mcp.tool
def get_clip_info(clip_id: str) -> dict:
    """Get full metadata for one clip: duration, size, fps, audio, history."""
    _get(clip_id)
    return _describe(clip_id)


@mcp.tool
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


@mcp.tool
def delete_clip(clip_id: str) -> dict:
    """Remove a clip from the registry and free its resources."""
    entry = _get(clip_id)
    try:
        entry.clip.close()
    except Exception:
        pass
    del _REGISTRY[clip_id]
    return {"deleted": clip_id, "remaining_clips": len(_REGISTRY)}


@mcp.resource("clip://{clip_id}/info", mime_type="application/json")
def clip_info_resource(clip_id: str) -> str:
    """JSON metadata for a registered clip."""
    _get(clip_id)
    return json.dumps(_describe(clip_id))


@mcp.resource("clip://{clip_id}/frame", mime_type="image/png")
def clip_frame_resource(clip_id: str) -> bytes:
    """PNG of the first frame (downscaled to 640px). Use preview_frame for other times."""
    entry = _get(clip_id, expect=("video", "image"))
    data, _, _ = _preview_png(entry.clip, 0.0, max_width=640)
    return data


# --------------------------------------------------------------------------- #
# Time operations
# --------------------------------------------------------------------------- #


@mcp.tool
def trim(clip_id: str, start_seconds: float,
         end_seconds: Optional[float] = None) -> dict:
    """Cut a clip to the range [start_seconds, end_seconds].

    Args:
        clip_id: Source clip (video or audio).
        start_seconds: Where the trimmed clip begins.
        end_seconds: Where it ends; omit to keep everything after start.

    Returns:
        Metadata of the NEW trimmed clip (new clip_id).
    """
    entry = _get(clip_id)
    if entry.clip.duration and start_seconds >= entry.clip.duration:
        raise ValueError(
            f"start_seconds ({start_seconds}) is beyond the clip duration "
            f"({entry.clip.duration:.2f}s).")
    new = entry.clip.subclipped(start_seconds, end_seconds)
    nid = _register(new, entry.kind, entry.label, entry,
                    f"trim({start_seconds}->{end_seconds})")
    return _describe(nid)


@mcp.tool
def concatenate(clip_ids: list[str], method: str = "compose") -> dict:
    """Join clips end-to-end, in the given order.

    All ids must be the same kind (all video/image, or all audio).

    Args:
        clip_ids: Two or more clip_ids in playback order.
        method: 'compose' (safe for mixed resolutions, pads smaller clips)
            or 'chain' (faster, requires identical sizes). Video only.
    """
    if len(clip_ids) < 2:
        raise ValueError("Provide at least two clip_ids to concatenate.")
    entries = [_get(cid) for cid in clip_ids]
    kinds = {("video" if e.kind == "image" else e.kind) for e in entries}
    if len(kinds) > 1:
        raise ValueError("Cannot concatenate video and audio together. "
                         "Use attach_audio to add sound to a video.")
    if kinds == {"audio"}:
        new = concatenate_audioclips([e.clip for e in entries])
        kind = "audio"
    else:
        new = concatenate_videoclips([e.clip for e in entries], method=method)
        kind = "video"
    nid = _register(new, kind, "concatenated", entries[0],
                    f"concatenate({clip_ids})")
    return _describe(nid)


@mcp.tool
def change_speed(clip_id: str, factor: float) -> dict:
    """Speed a clip up or down.

    Args:
        clip_id: Source clip.
        factor: 2.0 = double speed (half duration), 0.5 = slow motion.
    """
    if factor <= 0:
        raise ValueError("factor must be > 0.")
    entry = _get(clip_id)
    new = entry.clip.with_speed_scaled(factor)
    nid = _register(new, entry.kind, entry.label, entry, f"speed x{factor}")
    return _describe(nid)


@mcp.tool
def loop_clip(clip_id: str, n_times: Optional[int] = None,
              total_duration_seconds: Optional[float] = None) -> dict:
    """Repeat a clip. Give either n_times OR total_duration_seconds."""
    entry = _get(clip_id, expect=("video", "image"))
    if n_times is None and total_duration_seconds is None:
        raise ValueError("Provide n_times or total_duration_seconds.")
    new = entry.clip.with_effects(
        [vfx.Loop(n=n_times, duration=total_duration_seconds)])
    nid = _register(new, "video", entry.label, entry, "loop")
    return _describe(nid)


@mcp.tool
def reverse_clip(clip_id: str) -> dict:
    """Play a clip backwards (video, image sequence, or audio)."""
    entry = _get(clip_id)
    if not entry.clip.duration:
        raise ValueError("Clip has no duration; cannot reverse.")
    new = entry.clip.with_effects([vfx.TimeMirror()])
    nid = _register(new, entry.kind, entry.label, entry, "reverse")
    return _describe(nid)


@mcp.tool
def freeze(clip_id: str, freeze_duration_seconds: Optional[float] = None,
           total_duration_seconds: Optional[float] = None,
           time_seconds: float = 0.0, at_end: bool = False) -> dict:
    """Hold a frame, inserting freeze_duration extra seconds at that time.

    Give freeze_duration_seconds OR total_duration_seconds (clip + freeze).
    at_end=True freezes the last frame (ignores time_seconds).
    """
    if freeze_duration_seconds is None and total_duration_seconds is None:
        raise ValueError(
            "Provide freeze_duration_seconds or total_duration_seconds.")
    if freeze_duration_seconds is not None and freeze_duration_seconds <= 0:
        raise ValueError("freeze_duration_seconds must be > 0.")
    entry = _get(clip_id, expect=("video", "image"))
    duration = entry.clip.duration
    if not duration:
        raise ValueError("Clip has no duration; cannot freeze.")
    t: Any = time_seconds
    if at_end:
        t = max(0.0, duration - 1e-6)
    elif time_seconds < 0 or time_seconds > duration:
        raise ValueError(
            f"time_seconds {time_seconds} is outside [0, {duration}].")
    new = entry.clip.with_effects([
        vfx.Freeze(t=t, freeze_duration=freeze_duration_seconds,
                   total_duration=total_duration_seconds),
    ])
    nid = _register(new, "video", entry.label, entry, "freeze")
    return _describe(nid)


@mcp.tool
def crossfade(clip_ids: list[str], overlap_seconds: float) -> dict:
    """Join video clips with a dissolve instead of a hard cut.

    Each pair overlaps for overlap_seconds (CrossFadeOut + CrossFadeIn).
    Result duration is sum(durations) - (n-1)*overlap.
    """
    if len(clip_ids) < 2:
        raise ValueError("Provide at least two clip_ids to crossfade.")
    if overlap_seconds <= 0:
        raise ValueError(
            "overlap_seconds must be > 0. Use concatenate for a hard cut.")
    entries = [_get(cid, expect=("video", "image")) for cid in clip_ids]
    for entry in entries:
        dur = entry.clip.duration or 0
        if dur <= overlap_seconds:
            raise ValueError(
                f"Clip '{entry.label}' duration {dur}s must be greater than "
                f"overlap_seconds {overlap_seconds}s.")
    placed = []
    t = 0.0
    last = len(entries) - 1
    for i, entry in enumerate(entries):
        clip = entry.clip
        effects = []
        if i > 0:
            effects.append(vfx.CrossFadeIn(overlap_seconds))
        if i < last:
            effects.append(vfx.CrossFadeOut(overlap_seconds))
        if effects:
            clip = clip.with_effects(effects)
        placed.append(clip.with_start(t))
        t += entry.clip.duration - overlap_seconds
    new = CompositeVideoClip(placed)
    nid = _register(new, "video", "crossfade", entries[0],
                    f"crossfade(overlap={overlap_seconds})")
    return _describe(nid)


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


@mcp.tool
def resize(clip_id: str, width: Optional[int] = None,
           height: Optional[int] = None,
           scale: Optional[float] = None) -> dict:
    """Resize a video/image clip.

    Give exactly one of: scale (e.g. 0.5), width, height, or width+height.
    A single dimension preserves aspect ratio.
    """
    entry = _get(clip_id, expect=("video", "image"))
    if scale is not None:
        new = entry.clip.resized(scale)
        op = f"resize x{scale}"
    elif width and height:
        new = entry.clip.resized(new_size=(width, height))
        op = f"resize {width}x{height}"
    elif width:
        new = entry.clip.resized(width=width)
        op = f"resize w={width}"
    elif height:
        new = entry.clip.resized(height=height)
        op = f"resize h={height}"
    else:
        raise ValueError("Provide scale, width, height, or width+height.")
    nid = _register(new, entry.kind, entry.label, entry, op)
    return _describe(nid)


@mcp.tool
def crop(clip_id: str, x1: int, y1: int, x2: int, y2: int) -> dict:
    """Crop to the rectangle (x1, y1)-(x2, y2). Origin is the top-left."""
    entry = _get(clip_id, expect=("video", "image"))
    w, h = entry.clip.w, entry.clip.h
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        raise ValueError(f"Crop box ({x1},{y1})-({x2},{y2}) is outside the "
                         f"clip bounds {w}x{h}.")
    new = entry.clip.cropped(x1=x1, y1=y1, x2=x2, y2=y2)
    nid = _register(new, entry.kind, entry.label, entry,
                    f"crop({x1},{y1},{x2},{y2})")
    return _describe(nid)


@mcp.tool
def rotate(clip_id: str, degrees: float) -> dict:
    """Rotate counter-clockwise by the given degrees (use -90 for clockwise)."""
    entry = _get(clip_id, expect=("video", "image"))
    new = entry.clip.rotated(degrees, expand=True)
    nid = _register(new, entry.kind, entry.label, entry, f"rotate {degrees}")
    return _describe(nid)


@mcp.tool
def mirror(clip_id: str, axis: str = "horizontal") -> dict:
    """Flip a clip. axis: 'horizontal' (left-right) or 'vertical' (up-down)."""
    entry = _get(clip_id, expect=("video", "image"))
    effect = vfx.MirrorX() if axis == "horizontal" else vfx.MirrorY()
    new = entry.clip.with_effects([effect])
    nid = _register(new, entry.kind, entry.label, entry, f"mirror {axis}")
    return _describe(nid)


@mcp.tool
def even_size(clip_id: str) -> dict:
    """Crop 1px off odd width/height so H.264 export does not fail."""
    entry = _get(clip_id, expect=("video", "image"))
    new = entry.clip.with_effects([vfx.EvenSize()])
    nid = _register(new, entry.kind, entry.label, entry, "even_size")
    return _describe(nid)


# --------------------------------------------------------------------------- #
# Visual effects
# --------------------------------------------------------------------------- #


@mcp.tool
def fade(clip_id: str, fade_in_seconds: float = 0.0,
         fade_out_seconds: float = 0.0) -> dict:
    """Add fade-in and/or fade-out. Works for video (to black) and audio."""
    entry = _get(clip_id)
    effects = []
    if entry.kind == "audio":
        if fade_in_seconds:
            effects.append(afx.AudioFadeIn(fade_in_seconds))
        if fade_out_seconds:
            effects.append(afx.AudioFadeOut(fade_out_seconds))
    else:
        if fade_in_seconds:
            effects.append(vfx.FadeIn(fade_in_seconds))
        if fade_out_seconds:
            effects.append(vfx.FadeOut(fade_out_seconds))
    if not effects:
        raise ValueError("Give fade_in_seconds and/or fade_out_seconds > 0.")
    new = entry.clip.with_effects(effects)
    nid = _register(new, entry.kind, entry.label, entry,
                    f"fade(in={fade_in_seconds}, out={fade_out_seconds})")
    return _describe(nid)


@mcp.tool
def to_grayscale(clip_id: str) -> dict:
    """Convert a video/image clip to black and white."""
    entry = _get(clip_id, expect=("video", "image"))
    new = entry.clip.with_effects([vfx.BlackAndWhite()])
    nid = _register(new, entry.kind, entry.label, entry, "grayscale")
    return _describe(nid)


@mcp.tool
def adjust_colors(clip_id: str, brightness: float = 0.0,
                  contrast: float = 0.0) -> dict:
    """Adjust brightness/contrast.

    Args:
        brightness: -1.0 to 1.0 shift (0 = unchanged).
        contrast: -1.0 to 1.0 (0 = unchanged, positive increases contrast).
    """
    entry = _get(clip_id, expect=("video", "image"))
    new = entry.clip.with_effects(
        [vfx.LumContrast(lum=brightness * 255, contrast=contrast)])
    nid = _register(new, entry.kind, entry.label, entry,
                    f"colors(b={brightness}, c={contrast})")
    return _describe(nid)


@mcp.tool
def chroma_key(clip_id: str, color_rgb: list[int], threshold: float = 0.0,
               stiffness: float = 1.0) -> dict:
    """Mask a color in a video clip (chroma key / green screen).

    Args:
        clip_id: The video or image clip.
        color_rgb: The color to mask out as [R, G, B], e.g. [0, 255, 0] for green.
        threshold: Euclidean RGB distance tolerance. 0 = exact color only;
            try 20–100 for typical green-screen spill.
        stiffness: Edge sharpness. Higher = harder edges; lower = softer.
            MoviePy default is 1.0.
    """
    entry = _get(clip_id, expect=("video", "image"))
    if len(color_rgb) != 3:
        raise ValueError(
            f"color_rgb must be [R, G, B] with 3 ints, got {color_rgb!r}.")
    new = entry.clip.with_effects(
        [vfx.MaskColor(color=tuple(color_rgb), threshold=threshold,
                       stiffness=stiffness)])
    nid = _register(new, entry.kind, entry.label, entry,
                    f"chroma_key(rgb={color_rgb}, t={threshold}, s={stiffness})")
    return _describe(nid)


@mcp.tool
def invert_colors(clip_id: str) -> dict:
    """Invert colors (negative): black becomes white, etc."""
    entry = _get(clip_id, expect=("video", "image"))
    new = entry.clip.with_effects([vfx.InvertColors()])
    nid = _register(new, entry.kind, entry.label, entry, "invert_colors")
    return _describe(nid)


@mcp.tool
def gamma_correct(clip_id: str, gamma: float) -> dict:
    """Apply gamma correction. Typical values ~0.5–2.0; gamma must be > 0."""
    if gamma <= 0:
        raise ValueError(f"gamma must be > 0, got {gamma}.")
    entry = _get(clip_id, expect=("video", "image"))
    new = entry.clip.with_effects([vfx.GammaCorrection(gamma)])
    nid = _register(new, entry.kind, entry.label, entry, f"gamma({gamma})")
    return _describe(nid)


@mcp.tool
def multiply_color(clip_id: str, factor: float) -> dict:
    """Multiply RGB by factor: <1 darkens, >1 brightens. factor must be > 0."""
    if factor <= 0:
        raise ValueError(f"factor must be > 0, got {factor}.")
    entry = _get(clip_id, expect=("video", "image"))
    new = entry.clip.with_effects([vfx.MultiplyColor(factor)])
    nid = _register(new, entry.kind, entry.label, entry,
                    f"multiply_color({factor})")
    return _describe(nid)


@mcp.tool
def add_margin(clip_id: str, margin_size: Optional[int] = None,
               left: int = 0, right: int = 0, top: int = 0, bottom: int = 0,
               color_rgb: Optional[list[int]] = None) -> dict:
    """Add a colored border/margin around a video or image clip.

    Provide ``margin_size`` for equal margins on all sides, or set individual
    ``left``/``right``/``top``/``bottom`` pixel values (at least one > 0).
    """
    entry = _get(clip_id, expect=("video", "image"))
    color = color_rgb if color_rgb is not None else [0, 0, 0]
    if len(color) != 3:
        raise ValueError(
            f"color_rgb must be [R, G, B] with 3 ints, got {color!r}.")
    if margin_size is None and not (left or right or top or bottom):
        raise ValueError(
            "Provide margin_size or at least one of left/right/top/bottom > 0.")
    new = entry.clip.with_effects([
        vfx.Margin(margin_size=margin_size, left=left, right=right,
                   top=top, bottom=bottom, color=tuple(color))])
    op = (f"margin({margin_size})" if margin_size is not None
          else f"margin(L{left} R{right} T{top} B{bottom})")
    nid = _register(new, entry.kind, entry.label, entry, op)
    return _describe(nid)


@mcp.tool
def painting(clip_id: str, saturation: float = 1.4,
             black: float = 0.006) -> dict:
    """Stylize a clip to look like a painting.

    Args:
        saturation: How flashy the colors are (higher = more saturated).
        black: Amount of black contour (higher = stronger outlines).
    """
    entry = _get(clip_id, expect=("video", "image"))
    new = _ensure_uint8(entry.clip).with_effects(
        [vfx.Painting(saturation=saturation, black=black)])
    nid = _register(new, entry.kind, entry.label, entry,
                    f"painting(sat={saturation}, black={black})")
    return _describe(nid)


@mcp.tool
def blur(clip_id: str, radius: float = 2.0) -> dict:
    """Gaussian blur a video/image clip. radius >= 0 (Pillow)."""
    if radius < 0:
        raise ValueError(f"radius must be >= 0, got {radius}.")
    from PIL import ImageFilter
    entry = _get(clip_id, expect=("video", "image"))
    new = _pil_filter(
        entry.clip, lambda img: img.filter(ImageFilter.GaussianBlur(radius)))
    nid = _register(new, entry.kind, entry.label, entry, f"blur({radius})")
    return _describe(nid)


@mcp.tool
def sharpen(clip_id: str, percent: int = 150, radius: float = 2.0,
            threshold: int = 3) -> dict:
    """Unsharp-mask sharpen a video/image clip (Pillow).

    Args:
        percent: Strength of sharpening (100 = moderate, 150 = default).
        radius: Blur radius used by the unsharp mask.
        threshold: Minimum brightness change to sharpen (0–255).
    """
    from PIL import ImageFilter
    entry = _get(clip_id, expect=("video", "image"))
    new = _pil_filter(
        entry.clip,
        lambda img: img.filter(
            ImageFilter.UnsharpMask(radius=radius, percent=percent,
                                    threshold=threshold)))
    nid = _register(new, entry.kind, entry.label, entry,
                    f"sharpen(p={percent}, r={radius})")
    return _describe(nid)


@mcp.tool
def set_opacity(clip_id: str, opacity: float) -> dict:
    """Set clip opacity for compositing. 0.0 = invisible, 1.0 = fully opaque."""
    if not 0.0 <= opacity <= 1.0:
        raise ValueError(f"opacity must be between 0.0 and 1.0, got {opacity}.")
    entry = _get(clip_id, expect=("video", "image"))
    new = entry.clip.with_opacity(opacity)
    nid = _register(new, entry.kind, entry.label, entry, f"opacity({opacity})")
    return _describe(nid)


@mcp.tool
def freeze_region(clip_id: str, x1: int, y1: int, x2: int, y2: int,
                  time_seconds: float = 0.0) -> dict:
    """Freeze pixels inside (x1,y1)-(x2,y2) while the rest keeps playing."""
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Region must have x2 > x1 and y2 > y1.")
    entry = _get(clip_id, expect=("video", "image"))
    duration = entry.clip.duration or 0
    if time_seconds < 0 or (duration and time_seconds > duration):
        raise ValueError(
            f"time_seconds {time_seconds} is outside [0, {duration}].")
    new = entry.clip.with_effects([
        vfx.FreezeRegion(t=time_seconds, region=(x1, y1, x2, y2)),
    ])
    nid = _register(new, "video", entry.label, entry, "freeze_region")
    return _describe(nid)


@mcp.tool
def slide_in(clip_id: str, duration_seconds: float,
             side: str = "left") -> dict:
    """Slide the clip in from one side. Use with overlay_clip or concatenate.

    side: 'top', 'bottom', 'left', or 'right'.
    """
    if side not in _SLIDE_SIDES:
        raise ValueError(f"side must be one of {_SLIDE_SIDES}, got '{side}'.")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be > 0.")
    entry = _get(clip_id, expect=("video", "image"))
    dur = entry.clip.duration or 0
    if dur and duration_seconds > dur:
        raise ValueError("slide duration cannot exceed clip duration.")
    slid = entry.clip.with_effects([vfx.SlideIn(duration_seconds, side)])
    new = CompositeVideoClip([slid])
    nid = _register(new, "video", entry.label, entry, f"slide_in({side})")
    return _describe(nid)


@mcp.tool
def slide_out(clip_id: str, duration_seconds: float,
              side: str = "right") -> dict:
    """Slide the clip out toward one side. Use with overlay_clip or concatenate.

    side: 'top', 'bottom', 'left', or 'right'.
    """
    if side not in _SLIDE_SIDES:
        raise ValueError(f"side must be one of {_SLIDE_SIDES}, got '{side}'.")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be > 0.")
    entry = _get(clip_id, expect=("video", "image"))
    dur = entry.clip.duration or 0
    if dur and duration_seconds > dur:
        raise ValueError("slide duration cannot exceed clip duration.")
    slid = entry.clip.with_effects([vfx.SlideOut(duration_seconds, side)])
    new = CompositeVideoClip([slid])
    nid = _register(new, "video", entry.label, entry, f"slide_out({side})")
    return _describe(nid)


@mcp.tool
def scroll(clip_id: str, x_speed: float = 0.0, y_speed: float = 0.0,
           width: Optional[int] = None, height: Optional[int] = None,
           x_start: float = 0.0, y_start: float = 0.0) -> dict:
    """Scroll a clip (end credits, pan). Speeds are pixels per second."""
    if x_speed == 0 and y_speed == 0:
        raise ValueError("Provide x_speed and/or y_speed (pixels per second).")
    entry = _get(clip_id, expect=("video", "image"))
    kwargs: dict[str, Any] = dict(
        x_speed=x_speed, y_speed=y_speed, x_start=x_start, y_start=y_start,
    )
    if width is not None:
        kwargs["w"] = width
    if height is not None:
        kwargs["h"] = height
    new = entry.clip.with_effects([vfx.Scroll(**kwargs)])
    nid = _register(new, "video", entry.label, entry,
                    f"scroll(x={x_speed}, y={y_speed})")
    return _describe(nid)


@mcp.tool
def ken_burns(clip_id: str, start_zoom: float = 1.0, end_zoom: float = 1.2,
              start_x: float = 0.5, start_y: float = 0.5,
              end_x: float = 0.5, end_y: float = 0.5) -> dict:
    """Slow zoom/pan (Ken Burns). Output size matches the source.

    Zoom must be >= 1. x/y are normalized focal points (0=left/top, 1=right/bottom).
    """
    if start_zoom < 1 or end_zoom < 1:
        raise ValueError("start_zoom and end_zoom must be >= 1 (1 = no zoom).")
    for name, value in (("start_x", start_x), ("start_y", start_y),
                        ("end_x", end_x), ("end_y", end_y)):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}.")
    entry = _get(clip_id, expect=("video", "image"))
    new = _ken_burns(entry.clip, start_zoom, end_zoom,
                     start_x, start_y, end_x, end_y)
    nid = _register(new, "video", entry.label, entry,
                    f"ken_burns({start_zoom}->{end_zoom})")
    return _describe(nid)


# --------------------------------------------------------------------------- #
# Audio operations
# --------------------------------------------------------------------------- #


@mcp.tool
def set_volume(clip_id: str, factor: float) -> dict:
    """Scale volume: 0.5 = half, 2.0 = double, 0 = silent.

    Works on audio clips and on the audio track of video clips.
    """
    entry = _get(clip_id)
    if entry.kind == "video" and entry.clip.audio is None:
        raise ValueError("This video has no audio track.")
    new = entry.clip.with_volume_scaled(factor)
    nid = _register(new, entry.kind, entry.label, entry, f"volume x{factor}")
    return _describe(nid)


@mcp.tool
def normalize_audio(clip_id: str) -> dict:
    """Normalize volume to 0 dB. Works on audio clips and video soundtracks."""
    entry = _get(clip_id)
    return _apply_audio_effect(entry, afx.AudioNormalize(), "normalize_audio")


@mcp.tool
def delay_audio(clip_id: str, offset_seconds: float = 0.2,
                n_repeats: int = 8, decay: float = 1.0) -> dict:
    """Echo: repeat the audio n_repeats times, offset_seconds apart.

    decay < 1 fades repeats. Works on audio clips and video soundtracks.
    """
    if offset_seconds <= 0:
        raise ValueError("offset_seconds must be > 0.")
    if n_repeats < 1:
        raise ValueError("n_repeats must be >= 1.")
    entry = _get(clip_id)
    return _apply_audio_effect(
        entry,
        afx.AudioDelay(offset=offset_seconds, n_repeats=n_repeats, decay=decay),
        f"delay_audio({offset_seconds}x{n_repeats})",
    )


@mcp.tool
def set_stereo_volume(clip_id: str, left: float = 1.0,
                      right: float = 1.0) -> dict:
    """Set left/right channel gain (pan). Needs a stereo track.

    1.0 = unchanged, 0 = mute that side. Works on audio or video-with-audio.
    """
    if left < 0 or right < 0:
        raise ValueError("left and right must be >= 0.")
    entry = _get(clip_id)
    try:
        return _apply_audio_effect(
            entry, afx.MultiplyStereoVolume(left=left, right=right),
            f"stereo_volume(L={left}, R={right})")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            f"set_stereo_volume needs a stereo audio track: {exc}"
        ) from exc


@mcp.tool
def extract_audio(clip_id: str) -> dict:
    """Pull the audio track out of a video as a new audio clip."""
    entry = _get(clip_id, expect=("video",))
    if entry.clip.audio is None:
        raise ValueError("This video has no audio track to extract.")
    nid = _register(entry.clip.audio, "audio", f"audio of {entry.label}",
                    entry, "extract_audio")
    return _describe(nid)


@mcp.tool
def remove_audio(clip_id: str) -> dict:
    """Return a muted copy of a video clip (audio track removed)."""
    entry = _get(clip_id, expect=("video",))
    new = entry.clip.without_audio()
    nid = _register(new, "video", entry.label, entry, "remove_audio")
    return _describe(nid)


@mcp.tool
def attach_audio(video_clip_id: str, audio_clip_id: str,
                 mix_with_existing: bool = False) -> dict:
    """Set or mix an audio track onto a video clip.

    Args:
        video_clip_id: The video to receive audio.
        audio_clip_id: The audio clip to attach.
        mix_with_existing: If True and the video already has audio, mix both;
            if False, replace the existing track.
    """
    ventry = _get(video_clip_id, expect=("video", "image"))
    aentry = _get(audio_clip_id, expect=("audio",))
    audio = aentry.clip
    if mix_with_existing and ventry.clip.audio is not None:
        audio = CompositeAudioClip([ventry.clip.audio, audio])
    if audio.duration and ventry.clip.duration and \
            audio.duration > ventry.clip.duration:
        audio = audio.subclipped(0, ventry.clip.duration)
    new = ventry.clip.with_audio(audio)
    nid = _register(new, "video", ventry.label, ventry, "attach_audio")
    return _describe(nid)


# --------------------------------------------------------------------------- #
# Compositing
# --------------------------------------------------------------------------- #


@mcp.tool
def overlay_clip(base_clip_id: str, overlay_clip_id: str,
                 position: str = "center",
                 x: Optional[int] = None, y: Optional[int] = None,
                 start_seconds: float = 0.0,
                 duration_seconds: Optional[float] = None,
                 opacity: float = 1.0) -> dict:
    """Place one clip on top of another (text, watermark, picture-in-picture).

    Args:
        base_clip_id: The background/base video.
        overlay_clip_id: The clip to draw on top (video, image, or text clip).
        position: One of 'center', 'top', 'bottom', 'left', 'right',
            'top-left', 'top-right', 'bottom-left', 'bottom-right'.
            Ignored if x and y are given.
        x: Optional exact x position in pixels (top-left of overlay).
        y: Optional exact y position in pixels.
        start_seconds: When the overlay appears on the base timeline.
        duration_seconds: How long it stays; defaults to overlay's duration.
        opacity: 0.0 (invisible) to 1.0 (opaque).
    """
    base = _get(base_clip_id, expect=("video", "image"))
    over_entry = _get(overlay_clip_id, expect=("video", "image"))
    over = over_entry.clip

    if duration_seconds is not None:
        over = over.with_duration(duration_seconds)
    if opacity < 1.0:
        over = over.with_opacity(opacity)
    over = over.with_start(start_seconds)

    over = _with_position(over, position, x, y)

    new = CompositeVideoClip([base.clip, over])
    nid = _register(new, "video", base.label, base,
                    f"overlay({over_entry.label} @ {position})")
    return _describe(nid)


@mcp.tool
def grid_clips(clip_ids: list[str], columns: int,
               bg_color_rgb: Optional[list[int]] = None) -> dict:
    """Lay clips out left-to-right, wrapping every ``columns`` (side-by-side / 2x2).

    Clips are resized to the first clip's size. len(clip_ids) must divide evenly
    by columns.
    """
    if columns < 1:
        raise ValueError("columns must be >= 1.")
    if len(clip_ids) < 2:
        raise ValueError("Provide at least two clip_ids.")
    if len(clip_ids) % columns != 0:
        raise ValueError(
            f"{len(clip_ids)} clips does not fill a grid of {columns} columns. "
            "Add clips or change columns.")
    entries = [_get(cid, expect=("video", "image")) for cid in clip_ids]
    target = (entries[0].clip.w, entries[0].clip.h)
    cells = []
    for entry in entries:
        clip = entry.clip
        if (clip.w, clip.h) != target:
            clip = clip.resized(new_size=target)
        cells.append(clip)
    rows = [cells[i:i + columns] for i in range(0, len(cells), columns)]
    kwargs: dict[str, Any] = {}
    if bg_color_rgb is not None:
        kwargs["bg_color"] = tuple(bg_color_rgb)
    new = clips_array(rows, **kwargs)
    nid = _register(new, "video", "grid", entries[0],
                    f"grid({len(rows)}x{columns})")
    return _describe(nid)


@mcp.tool
def add_subtitles(clip_id: str, srt_text: Optional[str] = None,
                  srt_path: Optional[str] = None,
                  font_size: int = 32, color: str = "white",
                  font: Optional[str] = None,
                  stroke_color: str = "black", stroke_width: float = 2,
                  position: str = "bottom") -> dict:
    """Burn SRT or WebVTT cues onto a video as timed captions."""
    if bool(srt_text) == bool(srt_path):
        raise ValueError("Pass exactly one of srt_text or srt_path.")
    if srt_path:
        path = _check_path(srt_path)
        with open(path, encoding="utf-8") as fh:
            srt_text = fh.read()
    cues = _parse_subtitles(srt_text or "")
    base = _get(clip_id, expect=("video", "image"))
    base_dur = base.clip.duration or 0
    wrap_w = max(1, int(base.clip.w * 0.9))
    layers = [base.clip]
    for start, end, body in cues:
        if start >= base_dur:
            continue
        end = min(end, base_dur) if base_dur else end
        if end <= start:
            continue
        caption = _make_text_clip(
            text=body, font_size=font_size, color=color, font=font,
            duration_seconds=end - start, stroke_color=stroke_color,
            stroke_width=stroke_width, text_align="center", width=wrap_w,
            method="caption",
        )
        caption = _with_position(caption, position).with_start(start)
        layers.append(caption)
    if len(layers) == 1:
        raise ValueError("No cues fall within the clip duration.")
    new = CompositeVideoClip(layers)
    if base_dur:
        new = new.with_duration(base_dur)
    nid = _register(new, "video", base.label, base,
                    f"subtitles({len(layers) - 1} cues)")
    return _describe(nid)


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #


@mcp.tool
def save_frame(clip_id: str, output_path: str,
               time_seconds: float = 0.0) -> dict:
    """Save a single frame of a video as an image (png/jpg).

    Writes a file. Prefer ``preview_frame`` when the model needs to see
    the edit without touching disk.
    """
    entry = _get(clip_id, expect=("video", "image"))
    output_path = os.path.expanduser(output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    entry.clip.save_frame(output_path, t=time_seconds)
    return {"saved": output_path, "time_seconds": time_seconds,
            "clip_id": clip_id}


@mcp.tool
def export_image(clip_id: str, output_path: str,
                 time_seconds: float = 0.0) -> dict:
    """Export a still image file from a video or image clip.

    Args:
        clip_id: The video or image clip.
        output_path: Destination path; extension must be .png, .jpg, .jpeg, or .webp.
        time_seconds: Frame time for video clips (ignored for still images).
    """
    entry = _get(clip_id, expect=("video", "image"))
    output_path = os.path.expanduser(output_path)
    ext = os.path.splitext(output_path)[1].lower()
    allowed = {".png", ".jpg", ".jpeg", ".webp"}
    if ext not in allowed:
        raise ValueError(
            f"export_image requires extension in {sorted(allowed)}, got '{ext}'. "
            "Use save_frame for other formats or export_clip for video/audio.")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    entry.clip.save_frame(output_path, t=time_seconds)
    size = os.path.getsize(output_path)
    return {"exported": output_path, "file_size_bytes": size,
            "time_seconds": time_seconds, "clip_id": clip_id,
            "history": entry.history}


@mcp.tool
async def export_clip(clip_id: str, output_path: str, ctx: Context,
                      fps: Optional[float] = None,
                      codec: Optional[str] = None,
                      bitrate: Optional[str] = None) -> dict:
    """Render a clip to disk. THIS is the step that writes a file.

    Args:
        clip_id: The clip to render.
        output_path: Destination path. Extension picks the container:
            .mp4/.webm/.gif for video, .mp3/.wav/.ogg for audio.
        fps: Frames per second for video (defaults to source fps or 24).
        codec: Optional codec override (e.g. 'libx264', 'libvpx').
        bitrate: Optional bitrate (e.g. '4000k').

    Returns:
        The written path and final file size.
    """
    entry = _get(clip_id)
    output_path = os.path.expanduser(output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    await ctx.info(f"Exporting {clip_id} to {output_path}")
    await ctx.report_progress(0, None, "starting export")
    report = _bridge_progress(ctx)
    logger = _moviepy_logger(report)
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None,
        lambda: _write_clip(entry, output_path, fps, codec, bitrate, logger),
    )
    await ctx.report_progress(1, 1, "export complete")

    size = os.path.getsize(output_path)
    return {"exported": output_path, "file_size_bytes": size,
            "clip_id": clip_id, "history": entry.history}


def main() -> None:
    """Entry point for the `moviepy-mcp` console script (stdio transport)."""
    mcp.run()


if __name__ == "__main__":
    main()
