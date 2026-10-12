"""Frame, audio, subtitle, and file helpers shared by the tools."""

from __future__ import annotations

import asyncio
import base64
import binascii
import io
import os
import re
import tempfile
import uuid
from collections.abc import Callable
from typing import Any, Optional
from urllib.parse import urlparse

from fastmcp import Context
from moviepy import TextClip

from moviepy_mcp.paths import resolve_read
from moviepy_mcp.registry import ClipEntry

_POSITIONS = {
    "center": "center", "top": ("center", "top"),
    "bottom": ("center", "bottom"), "left": ("left", "center"),
    "right": ("right", "center"), "top-left": ("left", "top"),
    "top-right": ("right", "top"), "bottom-left": ("left", "bottom"),
    "bottom-right": ("right", "bottom"),
}

_SLIDE_SIDES = ("top", "bottom", "left", "right")

_TS_RE = re.compile(r"(\d{1,2}):(\d{2}):(\d{2})[.,](\d{1,3})")

_UPLOAD_DIR_NAME = "moviepy-mcp-uploads"

_MAX_INLINE_BYTES = 20 * 1024 * 1024

_STILL_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v", ".flv", ".wmv",
               ".gif"}

_AUDIO_EXTS = {".mp3", ".wav", ".aac", ".ogg", ".m4a", ".flac", ".wma"}

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}

_YT_HOSTS = ("youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be",
             "www.youtu.be", "music.youtube.com")

_IG_HOSTS = ("instagram.com", "www.instagram.com")

def _detect_kind(ext: str) -> Optional[str]:
    if ext in _VIDEO_EXTS:
        return "video"
    if ext in _AUDIO_EXTS:
        return "audio"
    if ext in _IMAGE_EXTS:
        return "image"
    return None

def _default_media_dir(name: str) -> str:
    return os.path.join(tempfile.gettempdir(), name)

def _decode_base64(data_base64: str) -> bytes:
    if not data_base64:
        raise ValueError("data_base64 is required and must not be empty.")
    try:
        raw = base64.b64decode(data_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"data_base64 is not valid base64: {exc}") from exc
    if not raw:
        raise ValueError("Decoded file is empty.")
    return raw

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
    cancel: Any = None,
) -> Any:
    from proglog import ProgressBarLogger

    class Logger(ProgressBarLogger):
        def bars_callback(self, bar, attr, value, old_value=None):
            # shortcut: cancel is noticed on the next progress tick, not by killing ffmpeg.
            if cancel is not None and cancel.is_set():
                raise RuntimeError("Export cancelled.")
            if attr != "index":
                return
            total = (self.bars.get(bar) or {}).get("total")
            if not total:
                return
            report(float(value), float(total), str(bar))

    return Logger(min_time_interval=0.25)

def _write_clip(entry: ClipEntry, output_path: str, fps: Optional[float],
                codec: Optional[str], bitrate: Optional[str],
                logger: Any, preset: Optional[str] = None) -> None:
    from moviepy_mcp.presets import get_preset, loudnorm_args, prepare_clip

    spec = get_preset(preset) if preset else None
    clip = entry.clip
    kind = entry.kind
    if spec is not None:
        clip = prepare_clip(clip, kind, spec)
        if fps is None:
            fps = spec.fps
        codec = codec or spec.codec
        bitrate = bitrate or spec.bitrate
    ext = os.path.splitext(output_path)[1].lower()
    if kind == "audio" or (spec is not None and spec.audio_only):
        kwargs: dict[str, Any] = {"logger": logger}
        if bitrate:
            kwargs["bitrate"] = bitrate
        if spec is not None and spec.audio_codec:
            kwargs["codec"] = spec.audio_codec
        if spec is not None and spec.loudness_lufs is not None:
            kwargs["ffmpeg_params"] = loudnorm_args(spec.loudness_lufs)
        clip.write_audiofile(output_path, **kwargs)
        return
    if ext == ".gif":
        clip.write_gif(output_path, fps=fps or 12, logger=logger)
        return
    out_fps = fps or getattr(clip, "fps", None) or 24
    kwargs = {"fps": out_fps, "logger": logger}
    if codec:
        kwargs["codec"] = codec
    if bitrate:
        kwargs["bitrate"] = bitrate
    if spec is not None:
        if spec.audio_codec:
            kwargs["audio_codec"] = spec.audio_codec
        if spec.audio_bitrate:
            kwargs["audio_bitrate"] = spec.audio_bitrate
        if spec.pixel_format:
            kwargs["pixel_format"] = spec.pixel_format
        extra = ["-movflags", "+faststart"]
        if spec.loudness_lufs is not None:
            extra.extend(loudnorm_args(spec.loudness_lufs))
        kwargs["ffmpeg_params"] = extra
    clip.write_videofile(output_path, **kwargs)

def _last_frame_time(clip: Any) -> float:
    """A timestamp just before the end, safe to pass to get_frame/save_frame."""
    duration = getattr(clip, "duration", None)
    if not duration:
        raise ValueError("Clip has no duration; cannot locate the last frame.")
    return max(0.0, duration - 1e-6)

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
        kwargs["font"] = resolve_read(font)
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

def _rms_db(chunk: Any) -> float:
    """RMS loudness of a mono float sample array, in dBFS."""
    import math

    import numpy as np

    if chunk.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(chunk))))
    if rms <= 1e-9:
        return -120.0
    return 20.0 * math.log10(rms)

def _detect_silence(audio: Any, threshold_db: float, min_silence_seconds: float,
                    window_seconds: float = 0.05) -> list[tuple[float, float]]:
    """Return (start, end) intervals of near-silence >= min_silence_seconds."""
    duration = audio.duration or 0.0
    if duration <= 0:
        return []
    fps = 22050
    array = audio.to_soundarray(fps=fps)
    if array.ndim > 1:
        array = array.mean(axis=1)
    window = max(1, int(window_seconds * fps))
    n_windows = max(1, len(array) // window) if len(array) else 0
    flags = [
        _rms_db(array[i * window:(i + 1) * window]) < threshold_db
        for i in range(n_windows)
    ]
    intervals: list[tuple[float, float]] = []
    start_idx: Optional[int] = None
    for i, flag in enumerate(flags + [False]):
        if flag and start_idx is None:
            start_idx = i
        elif not flag and start_idx is not None:
            start_t = start_idx * window_seconds
            end_t = min(i * window_seconds, duration)
            if end_t - start_t >= min_silence_seconds:
                intervals.append((start_t, end_t))
            start_idx = None
    return intervals

