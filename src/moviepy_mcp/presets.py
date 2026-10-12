"""Named delivery presets. Encode settings plus a target frame and loudness."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class DeliveryPreset:
    name: str
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    codec: Optional[str] = None
    bitrate: Optional[str] = None
    audio_codec: Optional[str] = None
    audio_bitrate: Optional[str] = None
    pixel_format: Optional[str] = None
    loudness_lufs: Optional[float] = None
    audio_only: bool = False


PRESETS: dict[str, DeliveryPreset] = {
    "youtube_1080p": DeliveryPreset(
        name="youtube_1080p",
        width=1920,
        height=1080,
        fps=30,
        codec="libx264",
        bitrate="8M",
        audio_codec="aac",
        audio_bitrate="192k",
        pixel_format="yuv420p",
        loudness_lufs=-14,
    ),
    "reels_1080x1920": DeliveryPreset(
        name="reels_1080x1920",
        width=1080,
        height=1920,
        fps=30,
        codec="libx264",
        bitrate="6M",
        audio_codec="aac",
        audio_bitrate="192k",
        pixel_format="yuv420p",
        loudness_lufs=-14,
    ),
    "podcast_aac": DeliveryPreset(
        name="podcast_aac",
        audio_codec="aac",
        audio_bitrate="128k",
        loudness_lufs=-16,
        audio_only=True,
    ),
}


def get_preset(name: str) -> DeliveryPreset:
    preset = PRESETS.get(name)
    if preset is None:
        known = ", ".join(sorted(PRESETS))
        raise ValueError(f"Unknown preset '{name}'. Known presets: {known}.")
    return preset


def fit_cover(clip: Any, width: int, height: int) -> Any:
    """Center-crop and scale so the frame fills width x height."""
    src_w, src_h = clip.w, clip.h
    target_ratio = width / height
    src_ratio = src_w / src_h
    if src_ratio > target_ratio:
        new_w, new_h = round(src_h * target_ratio), src_h
    else:
        new_w, new_h = src_w, round(src_w / target_ratio)
    x1 = max(0, (src_w - new_w) // 2)
    y1 = max(0, (src_h - new_h) // 2)
    return (
        clip.cropped(x1=x1, y1=y1, x2=x1 + new_w, y2=y1 + new_h)
        .resized(new_size=(width, height))
    )


def prepare_clip(clip: Any, kind: str, preset: DeliveryPreset) -> Any:
    """Apply frame-size rules for a preset. Does not encode."""
    from moviepy import vfx

    if preset.audio_only:
        if kind != "audio":
            raise ValueError(
                f"Preset '{preset.name}' is audio-only. extract_audio first, "
                "or use a video preset."
            )
        return clip
    if kind == "audio":
        raise ValueError(
            f"Preset '{preset.name}' renders video. Use podcast_aac for audio."
        )
    if preset.width and preset.height and (clip.w, clip.h) != (preset.width, preset.height):
        clip = fit_cover(clip, preset.width, preset.height)
    return clip.with_effects([vfx.EvenSize()])


def loudnorm_args(lufs: float) -> list[str]:
    # shortcut: one-pass loudnorm, not a measured two-pass EBU R128 render.
    return ["-af", f"loudnorm=I={lufs}:TP=-1.5:LRA=11"]
