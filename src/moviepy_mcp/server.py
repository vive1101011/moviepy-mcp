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
3. Structured, self-describing results: every tool returns a dict with
   ``clip_id`` plus metadata (duration, size, fps, has_audio) so the
   calling model always knows the current state without extra calls.
4. Fail loudly with actionable messages: errors name the offending
   argument and suggest the fix.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urlparse

from fastmcp import FastMCP
from moviepy import (
    AudioFileClip,
    ColorClip,
    CompositeAudioClip,
    CompositeVideoClip,
    ImageClip,
    TextClip,
    VideoFileClip,
    afx,
    concatenate_audioclips,
    concatenate_videoclips,
    vfx,
)

mcp = FastMCP(
    name="moviepy",
    instructions=(
        "Video/audio editing server. Workflow: load media with load_video / "
        "load_audio / load_image, or fetch from YouTube/Instagram with "
        "download_video (returns a clip_id), transform it with the editing "
        "tools (each returns a NEW clip_id — always use the latest id for the "
        "next step), then write the result with export_clip. "
        "Use list_clips / get_clip_info to inspect state at any time."
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
        info["size"] = {"width": clip.w, "height": clip.h}
        info["fps"] = getattr(clip, "fps", None)
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
                         cookies_file: Optional[str] = None) -> tuple[str, dict]:
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
def download_video(url: str, output_dir: Optional[str] = None,
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
    path, info = _download_with_ytdlp(
        url, dest,
        cookies_from_browser=cookies_from_browser,
        cookies_file=cookies_file)
    title = info.get("title") or os.path.basename(path)
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
        label: Optional human-readable name.
    """
    kwargs: dict[str, Any] = dict(text=text, font_size=font_size, color=color)
    if bg_color:
        kwargs["bg_color"] = bg_color
    if font:
        kwargs["font"] = _check_path(font)
    clip = TextClip(**kwargs).with_duration(duration_seconds)
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
def delete_clip(clip_id: str) -> dict:
    """Remove a clip from the registry and free its resources."""
    entry = _get(clip_id)
    try:
        entry.clip.close()
    except Exception:
        pass
    del _REGISTRY[clip_id]
    return {"deleted": clip_id, "remaining_clips": len(_REGISTRY)}


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

    if x is not None and y is not None:
        over = over.with_position((x, y))
    else:
        mapping = {
            "center": "center", "top": ("center", "top"),
            "bottom": ("center", "bottom"), "left": ("left", "center"),
            "right": ("right", "center"), "top-left": ("left", "top"),
            "top-right": ("right", "top"), "bottom-left": ("left", "bottom"),
            "bottom-right": ("right", "bottom"),
        }
        if position not in mapping:
            raise ValueError(f"Unknown position '{position}'. "
                             f"Valid: {', '.join(mapping)}")
        over = over.with_position(mapping[position])

    new = CompositeVideoClip([base.clip, over])
    nid = _register(new, "video", base.label, base,
                    f"overlay({over_entry.label} @ {position})")
    return _describe(nid)


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #


@mcp.tool
def save_frame(clip_id: str, output_path: str,
               time_seconds: float = 0.0) -> dict:
    """Save a single frame of a video as an image (png/jpg).

    Useful for thumbnails or letting the user preview an edit.
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
def export_clip(clip_id: str, output_path: str, fps: Optional[float] = None,
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
    ext = os.path.splitext(output_path)[1].lower()

    if entry.kind == "audio":
        entry.clip.write_audiofile(output_path, bitrate=bitrate)
    elif ext == ".gif":
        entry.clip.write_gif(output_path, fps=fps or 12)
    else:
        clip = entry.clip
        out_fps = fps or getattr(clip, "fps", None) or 24
        kwargs: dict[str, Any] = {"fps": out_fps, "logger": None}
        if codec:
            kwargs["codec"] = codec
        if bitrate:
            kwargs["bitrate"] = bitrate
        clip.write_videofile(output_path, **kwargs)

    size = os.path.getsize(output_path)
    return {"exported": output_path, "file_size_bytes": size,
            "clip_id": clip_id, "history": entry.history}


def main() -> None:
    """Entry point for the `moviepy-mcp` console script (stdio transport)."""
    mcp.run()


if __name__ == "__main__":
    main()
