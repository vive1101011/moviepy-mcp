"""Geometry and image effects."""

from __future__ import annotations

import asyncio
import os
import uuid
from typing import Any, Optional

from moviepy import CompositeVideoClip, afx, vfx
from mcp.types import ToolAnnotations

from moviepy_mcp.paths import resolve_dir, resolve_read, resolve_write, workspace_dir
from moviepy_mcp.registry import _REGISTRY, _describe, _get, _register, current_ids, remove
from moviepy_mcp.runtime import LOCK, serialized
from moviepy_mcp.server import mcp
from moviepy_mcp.editing import _SLIDE_SIDES, _ensure_uint8, _ken_burns, _pil_filter

READ = ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False)
EDIT = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False)
DESTROY = ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=False)
NET = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=True)


@mcp.tool(tags={'core'}, annotations=EDIT)
@serialized
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

@mcp.tool(tags={'core'}, annotations=EDIT)
@serialized
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

@mcp.tool(tags={'advanced'}, annotations=EDIT)
@serialized
def rotate(clip_id: str, degrees: float) -> dict:
    """Rotate counter-clockwise by the given degrees (use -90 for clockwise)."""
    entry = _get(clip_id, expect=("video", "image"))
    new = entry.clip.rotated(degrees, expand=True)
    nid = _register(new, entry.kind, entry.label, entry, f"rotate {degrees}")
    return _describe(nid)

@mcp.tool(tags={'advanced'}, annotations=EDIT)
@serialized
def mirror(clip_id: str, axis: str = "horizontal") -> dict:
    """Flip a clip. axis: 'horizontal' (left-right) or 'vertical' (up-down)."""
    entry = _get(clip_id, expect=("video", "image"))
    effect = vfx.MirrorX() if axis == "horizontal" else vfx.MirrorY()
    new = entry.clip.with_effects([effect])
    nid = _register(new, entry.kind, entry.label, entry, f"mirror {axis}")
    return _describe(nid)

@mcp.tool(tags={'core'}, annotations=EDIT)
@serialized
def even_size(clip_id: str) -> dict:
    """Crop 1px off odd width/height so H.264 export does not fail."""
    entry = _get(clip_id, expect=("video", "image"))
    new = entry.clip.with_effects([vfx.EvenSize()])
    nid = _register(new, entry.kind, entry.label, entry, "even_size")
    return _describe(nid)

@mcp.tool(tags={'core'}, annotations=EDIT)
@serialized
def reframe(clip_id: str, width: int, height: int,
           mode: str = "crop") -> dict:
    """Change a clip's aspect ratio (e.g. 1080x1920 for Shorts/Reels).

    Args:
        clip_id: Source video/image clip.
        width: Target width in pixels.
        height: Target height in pixels.
        mode: 'crop' (center-crop to fill the target box, cropping off the
            sides/top/bottom that don't fit) or 'pad_blur' (shrink the whole
            frame to fit inside the target with no content lost, filling the
            remaining bars with a blurred, zoomed copy of the same frame).
    """
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be > 0.")
    if mode not in ("crop", "pad_blur"):
        raise ValueError(f"mode must be 'crop' or 'pad_blur', got '{mode}'.")
    entry = _get(clip_id, expect=("video", "image"))
    clip = entry.clip
    src_w, src_h = clip.w, clip.h
    target_ratio = width / height
    src_ratio = src_w / src_h

    if mode == "crop":
        if src_ratio > target_ratio:
            new_w, new_h = round(src_h * target_ratio), src_h
        else:
            new_w, new_h = src_w, round(src_w / target_ratio)
        x1 = (src_w - new_w) // 2
        y1 = (src_h - new_h) // 2
        new = clip.cropped(x1=x1, y1=y1, x2=x1 + new_w, y2=y1 + new_h) \
                  .resized(new_size=(width, height))
        kind = entry.kind
    else:
        from PIL import ImageFilter
        fit_scale = min(width / src_w, height / src_h)
        fg = clip.resized(new_size=(round(src_w * fit_scale),
                                    round(src_h * fit_scale))) \
                  .with_position("center")
        cover_scale = max(width / src_w, height / src_h)
        bg = clip.resized(new_size=(round(src_w * cover_scale),
                                    round(src_h * cover_scale)))
        bx1 = (bg.w - width) // 2
        by1 = (bg.h - height) // 2
        bg = bg.cropped(x1=bx1, y1=by1, x2=bx1 + width, y2=by1 + height)
        bg = _pil_filter(_ensure_uint8(bg),
                         lambda img: img.filter(ImageFilter.GaussianBlur(25)))
        new = CompositeVideoClip([bg, fg], size=(width, height))
        kind = "video"
    nid = _register(new, kind, entry.label, entry,
                    f"reframe({width}x{height}, {mode})")
    return _describe(nid)

@mcp.tool(tags={'core'}, annotations=EDIT)
@serialized
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

@mcp.tool(tags={'advanced'}, annotations=EDIT)
@serialized
def to_grayscale(clip_id: str) -> dict:
    """Convert a video/image clip to black and white."""
    entry = _get(clip_id, expect=("video", "image"))
    new = entry.clip.with_effects([vfx.BlackAndWhite()])
    nid = _register(new, entry.kind, entry.label, entry, "grayscale")
    return _describe(nid)

@mcp.tool(tags={'advanced'}, annotations=EDIT)
@serialized
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

@mcp.tool(tags={'advanced'}, annotations=EDIT)
@serialized
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

@mcp.tool(tags={'advanced'}, annotations=EDIT)
@serialized
def invert_colors(clip_id: str) -> dict:
    """Invert colors (negative): black becomes white, etc."""
    entry = _get(clip_id, expect=("video", "image"))
    new = entry.clip.with_effects([vfx.InvertColors()])
    nid = _register(new, entry.kind, entry.label, entry, "invert_colors")
    return _describe(nid)

@mcp.tool(tags={'advanced'}, annotations=EDIT)
@serialized
def gamma_correct(clip_id: str, gamma: float) -> dict:
    """Apply gamma correction. Typical values ~0.5–2.0; gamma must be > 0."""
    if gamma <= 0:
        raise ValueError(f"gamma must be > 0, got {gamma}.")
    entry = _get(clip_id, expect=("video", "image"))
    new = entry.clip.with_effects([vfx.GammaCorrection(gamma)])
    nid = _register(new, entry.kind, entry.label, entry, f"gamma({gamma})")
    return _describe(nid)

@mcp.tool(tags={'advanced'}, annotations=EDIT)
@serialized
def multiply_color(clip_id: str, factor: float) -> dict:
    """Multiply RGB by factor: <1 darkens, >1 brightens. factor must be > 0."""
    if factor <= 0:
        raise ValueError(f"factor must be > 0, got {factor}.")
    entry = _get(clip_id, expect=("video", "image"))
    new = entry.clip.with_effects([vfx.MultiplyColor(factor)])
    nid = _register(new, entry.kind, entry.label, entry,
                    f"multiply_color({factor})")
    return _describe(nid)

@mcp.tool(tags={'advanced'}, annotations=EDIT)
@serialized
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

@mcp.tool(tags={'advanced'}, annotations=EDIT)
@serialized
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

@mcp.tool(tags={'advanced'}, annotations=EDIT)
@serialized
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

@mcp.tool(tags={'advanced'}, annotations=EDIT)
@serialized
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

@mcp.tool(tags={'advanced'}, annotations=EDIT)
@serialized
def set_opacity(clip_id: str, opacity: float) -> dict:
    """Set clip opacity for compositing. 0.0 = invisible, 1.0 = fully opaque."""
    if not 0.0 <= opacity <= 1.0:
        raise ValueError(f"opacity must be between 0.0 and 1.0, got {opacity}.")
    entry = _get(clip_id, expect=("video", "image"))
    new = entry.clip.with_opacity(opacity)
    nid = _register(new, entry.kind, entry.label, entry, f"opacity({opacity})")
    return _describe(nid)

@mcp.tool(tags={'advanced'}, annotations=EDIT)
@serialized
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

@mcp.tool(tags={'advanced'}, annotations=EDIT)
@serialized
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

@mcp.tool(tags={'advanced'}, annotations=EDIT)
@serialized
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

@mcp.tool(tags={'advanced'}, annotations=EDIT)
@serialized
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

@mcp.tool(tags={'advanced'}, annotations=EDIT)
@serialized
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
