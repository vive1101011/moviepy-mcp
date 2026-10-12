"""Overlays, grids, and burned-in subtitles."""

from __future__ import annotations

import asyncio
import os
import uuid
from typing import Any, Optional

from moviepy import CompositeVideoClip, clips_array
from mcp.types import ToolAnnotations

from moviepy_mcp.paths import resolve_dir, resolve_read, resolve_write, workspace_dir
from moviepy_mcp.registry import _REGISTRY, _describe, _get, _register, current_ids, remove
from moviepy_mcp.runtime import LOCK, serialized
from moviepy_mcp.server import mcp
from moviepy_mcp.editing import _make_text_clip, _parse_subtitles, _with_position

READ = ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False)
EDIT = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False)
DESTROY = ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=False)
NET = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=True)


@mcp.tool(tags={'core'}, annotations=EDIT)
@serialized
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
                    f"overlay({over_entry.label} @ {position})",
                    parents=[base, over_entry])
    return _describe(nid)

@mcp.tool(tags={'core'}, annotations=EDIT)
@serialized
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
                    f"grid({len(rows)}x{columns})", parents=entries)
    return _describe(nid)

@mcp.tool(tags={'core'}, annotations=EDIT)
@serialized
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
        path = resolve_read(srt_path)
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
