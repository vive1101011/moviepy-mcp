"""Cuts, joins, and time effects."""

from __future__ import annotations

import asyncio
import os
import uuid
from typing import Any, Optional

from moviepy import CompositeVideoClip, concatenate_audioclips, concatenate_videoclips, vfx
from mcp.types import ToolAnnotations

from moviepy_mcp.paths import resolve_dir, resolve_read, resolve_write, workspace_dir
from moviepy_mcp.registry import _REGISTRY, _describe, _get, _register, current_ids, remove
from moviepy_mcp.runtime import LOCK, serialized
from moviepy_mcp.server import mcp

READ = ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False)
EDIT = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False)
DESTROY = ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=False)
NET = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=True)


@mcp.tool(tags={'core'}, annotations=EDIT)
@serialized
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

@mcp.tool(tags={'core'}, annotations=EDIT)
@serialized
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
                    f"concatenate({clip_ids})", parents=entries)
    return _describe(nid)

@mcp.tool(tags={'core'}, annotations=EDIT)
@serialized
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

@mcp.tool(tags={'advanced'}, annotations=EDIT)
@serialized
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

@mcp.tool(tags={'advanced'}, annotations=EDIT)
@serialized
def reverse_clip(clip_id: str) -> dict:
    """Play a clip backwards (video, image sequence, or audio)."""
    entry = _get(clip_id)
    if not entry.clip.duration:
        raise ValueError("Clip has no duration; cannot reverse.")
    new = entry.clip.with_effects([vfx.TimeMirror()])
    nid = _register(new, entry.kind, entry.label, entry, "reverse")
    return _describe(nid)

@mcp.tool(tags={'advanced'}, annotations=EDIT)
@serialized
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

@mcp.tool(tags={'core'}, annotations=EDIT)
@serialized
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
                    f"crossfade(overlap={overlap_seconds})", parents=entries)
    return _describe(nid)
