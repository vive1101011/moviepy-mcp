import numpy as np
import pytest
from fastmcp import Client
from moviepy import AudioClip

from moviepy_mcp.server import _parse_subtitles, _register, mcp


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


def _tone(duration=0.3) -> str:
    clip = AudioClip(
        lambda t: 0.2 * np.sin(2 * np.pi * 440 * t),
        duration=duration,
        fps=44100,
    )
    return _register(clip, "audio", "tone", None, "test")


@pytest.mark.asyncio
async def test_reverse_keeps_duration():
    async with Client(mcp) as client:
        created = await _color(client, duration=1.0)
        out = _data(await client.call_tool("reverse_clip", {"clip_id": created["clip_id"]}))
        assert out["clip_id"] != created["clip_id"]
        assert out["duration_seconds"] == 1.0
        assert "reverse" in out["history"][-1]


@pytest.mark.asyncio
async def test_freeze_extends_duration():
    async with Client(mcp) as client:
        created = await _color(client, duration=1.0)
        out = _data(
            await client.call_tool(
                "freeze",
                {"clip_id": created["clip_id"], "freeze_duration_seconds": 0.5},
            )
        )
        assert out["duration_seconds"] == 1.5


@pytest.mark.asyncio
async def test_freeze_requires_duration():
    async with Client(mcp) as client:
        created = await _color(client)
        result = await client.call_tool(
            "freeze", {"clip_id": created["clip_id"]}, raise_on_error=False
        )
        assert result.is_error


@pytest.mark.asyncio
async def test_crossfade_overlap_duration():
    async with Client(mcp) as client:
        a = await _color(client, duration=1.0, color=[255, 0, 0])
        b = await _color(client, duration=1.0, color=[0, 0, 255])
        out = _data(
            await client.call_tool(
                "crossfade",
                {"clip_ids": [a["clip_id"], b["clip_id"]], "overlap_seconds": 0.25},
            )
        )
        assert out["duration_seconds"] == 1.75


@pytest.mark.asyncio
async def test_crossfade_rejects_overlap_longer_than_clip():
    async with Client(mcp) as client:
        a = await _color(client, duration=0.4)
        b = await _color(client, duration=0.4)
        result = await client.call_tool(
            "crossfade",
            {"clip_ids": [a["clip_id"], b["clip_id"]], "overlap_seconds": 0.5},
            raise_on_error=False,
        )
        assert result.is_error


@pytest.mark.asyncio
async def test_even_size_crops_odd_dimensions():
    async with Client(mcp) as client:
        created = await _color(client, width=33, height=31)
        out = _data(await client.call_tool("even_size", {"clip_id": created["clip_id"]}))
        assert out["size"] == {"width": 32, "height": 30}


@pytest.mark.asyncio
async def test_grid_clips_2x2():
    async with Client(mcp) as client:
        ids = [(await _color(client, duration=0.5))["clip_id"] for _ in range(4)]
        out = _data(
            await client.call_tool("grid_clips", {"clip_ids": ids, "columns": 2})
        )
        assert out["size"] == {"width": 64, "height": 64}
        assert out["duration_seconds"] == 0.5


@pytest.mark.asyncio
async def test_grid_clips_requires_full_rows():
    async with Client(mcp) as client:
        ids = [(await _color(client))["clip_id"] for _ in range(3)]
        result = await client.call_tool(
            "grid_clips",
            {"clip_ids": ids, "columns": 2},
            raise_on_error=False,
        )
        assert result.is_error


@pytest.mark.asyncio
async def test_ken_burns_preserves_size_and_duration():
    async with Client(mcp) as client:
        created = await _color(client, width=64, height=32, duration=0.5)
        out = _data(
            await client.call_tool(
                "ken_burns",
                {
                    "clip_id": created["clip_id"],
                    "start_zoom": 1.0,
                    "end_zoom": 1.4,
                    "end_x": 0.8,
                    "end_y": 0.2,
                },
            )
        )
        assert out["size"] == {"width": 64, "height": 32}
        assert out["duration_seconds"] == 0.5
        frame = await client.call_tool(
            "preview_frame",
            {"clip_id": out["clip_id"], "time_seconds": 0.25, "max_width": 0},
        )
        assert not frame.is_error


@pytest.mark.asyncio
async def test_ken_burns_rejects_zoom_below_one():
    async with Client(mcp) as client:
        created = await _color(client)
        result = await client.call_tool(
            "ken_burns",
            {"clip_id": created["clip_id"], "start_zoom": 0.5},
            raise_on_error=False,
        )
        assert result.is_error


@pytest.mark.asyncio
async def test_scroll_and_slides_keep_duration():
    async with Client(mcp) as client:
        created = await _color(client, width=64, height=64, duration=1.0)
        scrolled = _data(
            await client.call_tool(
                "scroll",
                {"clip_id": created["clip_id"], "y_speed": 40, "height": 32},
            )
        )
        assert scrolled["duration_seconds"] == 1.0
        slid = _data(
            await client.call_tool(
                "slide_in",
                {
                    "clip_id": created["clip_id"],
                    "duration_seconds": 0.25,
                    "side": "left",
                },
            )
        )
        assert slid["duration_seconds"] == 1.0
        out = _data(
            await client.call_tool(
                "slide_out",
                {
                    "clip_id": created["clip_id"],
                    "duration_seconds": 0.25,
                    "side": "right",
                },
            )
        )
        assert out["duration_seconds"] == 1.0


@pytest.mark.asyncio
async def test_freeze_region_keeps_duration():
    async with Client(mcp) as client:
        created = await _color(client, width=32, height=32, duration=0.5)
        out = _data(
            await client.call_tool(
                "freeze_region",
                {
                    "clip_id": created["clip_id"],
                    "x1": 0,
                    "y1": 0,
                    "x2": 16,
                    "y2": 16,
                    "time_seconds": 0.0,
                },
            )
        )
        assert out["duration_seconds"] == 0.5
        assert out["size"] == {"width": 32, "height": 32}


@pytest.mark.asyncio
async def test_audio_normalize_delay_stereo():
    async with Client(mcp) as client:
        cid = _tone()
        norm = _data(await client.call_tool("normalize_audio", {"clip_id": cid}))
        assert norm["kind"] == "audio"
        delayed = _data(
            await client.call_tool(
                "delay_audio",
                {"clip_id": cid, "offset_seconds": 0.05, "n_repeats": 2, "decay": 0.5},
            )
        )
        assert delayed["kind"] == "audio"
        stereo = _data(
            await client.call_tool(
                "set_stereo_volume",
                {"clip_id": cid, "left": 0.5, "right": 1.0},
            )
        )
        assert stereo["kind"] == "audio"


@pytest.mark.asyncio
async def test_normalize_audio_rejects_silent_video():
    async with Client(mcp) as client:
        created = await _color(client)
        result = await client.call_tool(
            "normalize_audio",
            {"clip_id": created["clip_id"]},
            raise_on_error=False,
        )
        assert result.is_error


def test_parse_subtitles_srt_and_vtt():
    srt = (
        "1\n00:00:01,000 --> 00:00:04,000\nHello world\n\n"
        "2\n00:00:05,500 --> 00:00:06,000\nSecond"
    )
    cues = _parse_subtitles(srt)
    assert cues == [(1.0, 4.0, "Hello world"), (5.5, 6.0, "Second")]

    vtt = "WEBVTT\n\n00:00:00.000 --> 00:00:01.250\n<c>Hi</c>"
    assert _parse_subtitles(vtt) == [(0.0, 1.25, "Hi")]


def test_parse_subtitles_empty_is_error():
    with pytest.raises(ValueError, match="No subtitle cues"):
        _parse_subtitles("WEBVTT\n\nNOTE skip me")


@pytest.mark.asyncio
async def test_add_subtitles_keeps_base_duration():
    async with Client(mcp) as client:
        created = await _color(client, width=320, height=180, duration=2.0)
        out = _data(
            await client.call_tool(
                "add_subtitles",
                {
                    "clip_id": created["clip_id"],
                    "srt_text": "1\n00:00:00,000 --> 00:00:01,000\nHello",
                    "font_size": 18,
                },
            )
        )
        assert out["duration_seconds"] == 2.0
        assert out["size"] == {"width": 320, "height": 180}
        assert "subtitles" in out["history"][-1]


@pytest.mark.asyncio
async def test_add_subtitles_requires_one_source():
    async with Client(mcp) as client:
        created = await _color(client)
        both = await client.call_tool(
            "add_subtitles",
            {
                "clip_id": created["clip_id"],
                "srt_text": "x",
                "srt_path": "subs.srt",
            },
            raise_on_error=False,
        )
        neither = await client.call_tool(
            "add_subtitles",
            {"clip_id": created["clip_id"]},
            raise_on_error=False,
        )
        assert both.is_error
        assert neither.is_error


@pytest.mark.asyncio
async def test_create_text_clip_caption_wrap():
    async with Client(mcp) as client:
        out = _data(
            await client.call_tool(
                "create_text_clip",
                {
                    "text": "Hello wrapping world",
                    "font_size": 20,
                    "width": 200,
                    "duration_seconds": 0.5,
                    "stroke_color": "black",
                    "stroke_width": 1,
                    "text_align": "center",
                },
            )
        )
        assert out["size"]["width"] == 200
        assert out["duration_seconds"] == 0.5
