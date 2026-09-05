import math
from pathlib import Path

import numpy as np
import pytest
from fastmcp import Client
from moviepy import AudioClip
from PIL import Image as PILImage

from moviepy_mcp.server import _register, mcp

pytestmark = pytest.mark.asyncio


def _data(result) -> dict:
    assert isinstance(result.data, dict)
    return result.data


async def _color(client, width=32, height=32, duration=1.0, color=None):
    return _data(
        await client.call_tool(
            "create_color_clip",
            {
                "width": width,
                "height": height,
                "color_rgb": color or [255, 0, 0],
                "duration_seconds": duration,
            },
        )
    )


def _tone(duration=0.5, freq=440.0, amplitude=0.5) -> str:
    clip = AudioClip(
        lambda t: amplitude * np.sin(2 * np.pi * freq * t),
        duration=duration,
        fps=44100,
    )
    return _register(clip, "audio", "tone", None, "test")


def _silence(duration=0.5) -> str:
    clip = AudioClip(lambda t: 0.0 * t, duration=duration, fps=44100)
    return _register(clip, "audio", "silence", None, "test")


# --------------------------------------------------------------------------- #
# mix_audio_tracks
# --------------------------------------------------------------------------- #


async def test_mix_audio_tracks_keeps_longest_duration():
    async with Client(mcp) as client:
        a = _tone(duration=0.5)
        b = _tone(duration=1.0)
        out = _data(
            await client.call_tool(
                "mix_audio_tracks", {"clip_ids": [a, b]}
            )
        )
        assert out["kind"] == "audio"
        assert out["duration_seconds"] == 1.0


async def test_mix_audio_tracks_applies_per_clip_volumes():
    async with Client(mcp) as client:
        a = _tone(duration=0.3)
        b = _tone(duration=0.3)
        out = _data(
            await client.call_tool(
                "mix_audio_tracks",
                {"clip_ids": [a, b], "volumes": [1.0, 0.2]},
            )
        )
        assert out["kind"] == "audio"


async def test_mix_audio_tracks_requires_two_clips():
    async with Client(mcp) as client:
        a = _tone()
        result = await client.call_tool(
            "mix_audio_tracks", {"clip_ids": [a]}, raise_on_error=False
        )
        assert result.is_error


async def test_mix_audio_tracks_rejects_mismatched_volumes_length():
    async with Client(mcp) as client:
        a, b = _tone(), _tone()
        result = await client.call_tool(
            "mix_audio_tracks",
            {"clip_ids": [a, b], "volumes": [1.0]},
            raise_on_error=False,
        )
        assert result.is_error


async def test_mix_audio_tracks_rejects_video_input():
    async with Client(mcp) as client:
        v = await _color(client)
        a = _tone()
        result = await client.call_tool(
            "mix_audio_tracks",
            {"clip_ids": [v["clip_id"], a]},
            raise_on_error=False,
        )
        assert result.is_error


# --------------------------------------------------------------------------- #
# reframe
# --------------------------------------------------------------------------- #


async def test_reframe_crop_to_vertical():
    async with Client(mcp) as client:
        created = await _color(client, width=160, height=90)
        out = _data(
            await client.call_tool(
                "reframe",
                {"clip_id": created["clip_id"], "width": 90, "height": 160,
                 "mode": "crop"},
            )
        )
        assert out["size"] == {"width": 90, "height": 160}
        assert out["kind"] == "video"


async def test_reframe_pad_blur_to_vertical():
    async with Client(mcp) as client:
        created = await _color(client, width=160, height=90, duration=0.5)
        out = _data(
            await client.call_tool(
                "reframe",
                {"clip_id": created["clip_id"], "width": 90, "height": 160,
                 "mode": "pad_blur"},
            )
        )
        assert out["size"] == {"width": 90, "height": 160}
        assert out["duration_seconds"] == 0.5
        preview = await client.call_tool(
            "preview_frame",
            {"clip_id": out["clip_id"], "max_width": 0},
        )
        assert not preview.is_error


async def test_reframe_rejects_bad_mode():
    async with Client(mcp) as client:
        created = await _color(client)
        result = await client.call_tool(
            "reframe",
            {"clip_id": created["clip_id"], "width": 10, "height": 10,
             "mode": "stretch"},
            raise_on_error=False,
        )
        assert result.is_error


async def test_reframe_rejects_non_positive_dims():
    async with Client(mcp) as client:
        created = await _color(client)
        result = await client.call_tool(
            "reframe",
            {"clip_id": created["clip_id"], "width": 0, "height": 10},
            raise_on_error=False,
        )
        assert result.is_error


# --------------------------------------------------------------------------- #
# create_slideshow
# --------------------------------------------------------------------------- #


async def test_create_slideshow_hard_cuts(tmp_path: Path):
    paths = []
    for i, color in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
        p = tmp_path / f"slide{i}.png"
        PILImage.new("RGB", (40, 30), color).save(p)
        paths.append(str(p))

    async with Client(mcp) as client:
        out = _data(
            await client.call_tool(
                "create_slideshow",
                {"image_paths": paths, "duration_seconds": 0.5},
            )
        )
        assert out["duration_seconds"] == pytest.approx(1.5)
        assert out["size"] == {"width": 40, "height": 30}


async def test_create_slideshow_with_crossfade(tmp_path: Path):
    paths = []
    for i, color in enumerate([(255, 0, 0), (0, 255, 0)]):
        p = tmp_path / f"slide{i}.png"
        PILImage.new("RGB", (40, 30), color).save(p)
        paths.append(str(p))

    async with Client(mcp) as client:
        out = _data(
            await client.call_tool(
                "create_slideshow",
                {
                    "image_paths": paths,
                    "duration_seconds": 1.0,
                    "transition_seconds": 0.25,
                },
            )
        )
        assert out["duration_seconds"] == pytest.approx(1.75)


async def test_create_slideshow_with_audio_loops_short_track(tmp_path: Path):
    paths = []
    for i, color in enumerate([(255, 0, 0), (0, 255, 0)]):
        p = tmp_path / f"slide{i}.png"
        PILImage.new("RGB", (40, 30), color).save(p)
        paths.append(str(p))
    audio_id = _tone(duration=0.3)

    async with Client(mcp) as client:
        out = _data(
            await client.call_tool(
                "create_slideshow",
                {
                    "image_paths": paths,
                    "duration_seconds": 0.5,
                    "audio_clip_id": audio_id,
                },
            )
        )
        assert out["has_audio"] is True
        assert out["duration_seconds"] == pytest.approx(1.0)


async def test_create_slideshow_requires_two_images(tmp_path: Path):
    p = tmp_path / "only.png"
    PILImage.new("RGB", (10, 10), (0, 0, 0)).save(p)
    async with Client(mcp) as client:
        result = await client.call_tool(
            "create_slideshow",
            {"image_paths": [str(p)]},
            raise_on_error=False,
        )
        assert result.is_error


async def test_create_slideshow_rejects_transition_ge_duration(tmp_path: Path):
    paths = []
    for i in range(2):
        p = tmp_path / f"s{i}.png"
        PILImage.new("RGB", (10, 10), (0, 0, 0)).save(p)
        paths.append(str(p))
    async with Client(mcp) as client:
        result = await client.call_tool(
            "create_slideshow",
            {
                "image_paths": paths,
                "duration_seconds": 0.5,
                "transition_seconds": 0.5,
            },
            raise_on_error=False,
        )
        assert result.is_error


# --------------------------------------------------------------------------- #
# remove_silence
# --------------------------------------------------------------------------- #


def _tone_with_silence_gap() -> str:
    """0.4s tone, 0.8s near-silence, 0.4s tone."""
    total = 1.6

    def make_frame(t):
        scalar = np.ndim(t) == 0
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        loud = 0.5 * np.sin(2 * np.pi * 440 * t_arr)
        gap = (t_arr >= 0.4) & (t_arr < 1.2)
        result = np.where(gap, 0.0, loud)
        return result[0] if scalar else result

    clip = AudioClip(make_frame, duration=total, fps=44100)
    return _register(clip, "audio", "tone_with_gap", None, "test")


async def test_remove_silence_shortens_audio_clip():
    async with Client(mcp) as client:
        cid = _tone_with_silence_gap()
        out = _data(
            await client.call_tool(
                "remove_silence",
                {
                    "clip_id": cid,
                    "silence_threshold_db": -30.0,
                    "min_silence_seconds": 0.3,
                    "padding_seconds": 0.05,
                },
            )
        )
        assert out["kind"] == "audio"
        # gap was ~0.8s minus 2x padding; result should be well under original 1.6s
        assert out["duration_seconds"] < 1.2
        assert "remove_silence" in out["history"][-1]


async def test_remove_silence_on_video_keeps_kind_video():
    async with Client(mcp) as client:
        video = await _color(client, duration=1.6)
        audio_id = _tone_with_silence_gap()
        with_audio = _data(
            await client.call_tool(
                "attach_audio",
                {"video_clip_id": video["clip_id"], "audio_clip_id": audio_id},
            )
        )
        out = _data(
            await client.call_tool(
                "remove_silence",
                {
                    "clip_id": with_audio["clip_id"],
                    "silence_threshold_db": -30.0,
                    "min_silence_seconds": 0.3,
                    "padding_seconds": 0.05,
                },
            )
        )
        assert out["kind"] == "video"
        assert out["duration_seconds"] < 1.2


async def test_remove_silence_requires_audio_track():
    async with Client(mcp) as client:
        created = await _color(client)
        result = await client.call_tool(
            "remove_silence", {"clip_id": created["clip_id"]},
            raise_on_error=False,
        )
        assert result.is_error


async def test_remove_silence_raises_when_nothing_found():
    async with Client(mcp) as client:
        cid = _tone(duration=0.5)
        result = await client.call_tool(
            "remove_silence",
            {"clip_id": cid, "silence_threshold_db": -80.0},
            raise_on_error=False,
        )
        assert result.is_error


async def test_remove_silence_rejects_bad_args():
    async with Client(mcp) as client:
        cid = _tone(duration=0.5)
        result = await client.call_tool(
            "remove_silence", {"clip_id": cid, "min_silence_seconds": 0},
            raise_on_error=False,
        )
        assert result.is_error
