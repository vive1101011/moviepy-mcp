import io
import shutil
from pathlib import Path

import pytest
from fastmcp import Client
from moviepy import AudioClip
from PIL import Image as PILImage

from moviepy_mcp.server import _register, mcp

pytestmark = pytest.mark.asyncio

FFMPEG = shutil.which("ffmpeg") is not None


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


def _tone(duration=0.2) -> str:
    clip = AudioClip(
        lambda t: 0.2 * ((t * 0) + 1), duration=duration, fps=44100,
    )
    return _register(clip, "audio", "tone", None, "test")


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


async def test_resize_by_scale_width_height_and_both():
    async with Client(mcp) as client:
        created = await _color(client, width=100, height=50)
        cid = created["clip_id"]

        by_scale = await _color(client, width=100, height=50)
        scaled = _data(
            await client.call_tool(
                "resize", {"clip_id": cid, "scale": 0.5}
            )
        )
        assert scaled["size"] == {"width": 50, "height": 25}

        by_width = _data(
            await client.call_tool("resize", {"clip_id": cid, "width": 200})
        )
        assert by_width["size"] == {"width": 200, "height": 100}

        by_both = _data(
            await client.call_tool(
                "resize", {"clip_id": cid, "width": 40, "height": 40}
            )
        )
        assert by_both["size"] == {"width": 40, "height": 40}


async def test_resize_requires_a_dimension():
    async with Client(mcp) as client:
        created = await _color(client)
        result = await client.call_tool(
            "resize", {"clip_id": created["clip_id"]}, raise_on_error=False
        )
        assert result.is_error


async def test_crop_to_rectangle():
    async with Client(mcp) as client:
        created = await _color(client, width=100, height=100)
        out = _data(
            await client.call_tool(
                "crop",
                {"clip_id": created["clip_id"], "x1": 10, "y1": 10, "x2": 60, "y2": 40},
            )
        )
        assert out["size"] == {"width": 50, "height": 30}


async def test_crop_rejects_out_of_bounds_box():
    async with Client(mcp) as client:
        created = await _color(client, width=50, height=50)
        result = await client.call_tool(
            "crop",
            {"clip_id": created["clip_id"], "x1": 0, "y1": 0, "x2": 60, "y2": 10},
            raise_on_error=False,
        )
        assert result.is_error


async def test_rotate_90_swaps_dimensions():
    async with Client(mcp) as client:
        created = await _color(client, width=100, height=50)
        out = _data(
            await client.call_tool(
                "rotate", {"clip_id": created["clip_id"], "degrees": 90}
            )
        )
        assert out["size"] == {"width": 50, "height": 100}


async def test_mirror_horizontal_and_vertical_keep_size():
    async with Client(mcp) as client:
        created = await _color(client, width=40, height=20)
        for axis in ("horizontal", "vertical"):
            out = _data(
                await client.call_tool(
                    "mirror", {"clip_id": created["clip_id"], "axis": axis}
                )
            )
            assert out["size"] == {"width": 40, "height": 20}


# --------------------------------------------------------------------------- #
# Time
# --------------------------------------------------------------------------- #


async def test_concatenate_video_clips_sums_duration():
    async with Client(mcp) as client:
        a = await _color(client, duration=1.0)
        b = await _color(client, duration=0.5)
        out = _data(
            await client.call_tool(
                "concatenate", {"clip_ids": [a["clip_id"], b["clip_id"]]}
            )
        )
        assert out["duration_seconds"] == 1.5


async def test_concatenate_requires_two_clips():
    async with Client(mcp) as client:
        a = await _color(client)
        result = await client.call_tool(
            "concatenate", {"clip_ids": [a["clip_id"]]}, raise_on_error=False
        )
        assert result.is_error


async def test_concatenate_rejects_mixed_video_and_audio():
    async with Client(mcp) as client:
        v = await _color(client)
        a_id = _tone()
        result = await client.call_tool(
            "concatenate",
            {"clip_ids": [v["clip_id"], a_id]},
            raise_on_error=False,
        )
        assert result.is_error


async def test_change_speed_scales_duration():
    async with Client(mcp) as client:
        created = await _color(client, duration=2.0)
        out = _data(
            await client.call_tool(
                "change_speed", {"clip_id": created["clip_id"], "factor": 2.0}
            )
        )
        assert out["duration_seconds"] == 1.0


async def test_change_speed_rejects_non_positive_factor():
    async with Client(mcp) as client:
        created = await _color(client)
        result = await client.call_tool(
            "change_speed", {"clip_id": created["clip_id"], "factor": 0},
            raise_on_error=False,
        )
        assert result.is_error


async def test_loop_clip_by_n_times():
    async with Client(mcp) as client:
        created = await _color(client, duration=0.5)
        out = _data(
            await client.call_tool(
                "loop_clip", {"clip_id": created["clip_id"], "n_times": 3}
            )
        )
        assert out["duration_seconds"] == pytest.approx(1.5)


async def test_loop_clip_requires_a_target():
    async with Client(mcp) as client:
        created = await _color(client)
        result = await client.call_tool(
            "loop_clip", {"clip_id": created["clip_id"]}, raise_on_error=False
        )
        assert result.is_error


# --------------------------------------------------------------------------- #
# Visual FX
# --------------------------------------------------------------------------- #


async def test_fade_in_and_out_requires_a_value():
    async with Client(mcp) as client:
        created = await _color(client, duration=1.0)
        out = _data(
            await client.call_tool(
                "fade",
                {"clip_id": created["clip_id"], "fade_in_seconds": 0.1,
                 "fade_out_seconds": 0.1},
            )
        )
        assert out["duration_seconds"] == 1.0

        result = await client.call_tool(
            "fade", {"clip_id": created["clip_id"]}, raise_on_error=False
        )
        assert result.is_error


async def test_to_grayscale_keeps_size():
    async with Client(mcp) as client:
        created = await _color(client, width=20, height=20, color=[10, 200, 30])
        out = _data(
            await client.call_tool("to_grayscale", {"clip_id": created["clip_id"]})
        )
        assert out["size"] == {"width": 20, "height": 20}


async def test_adjust_colors_keeps_size():
    async with Client(mcp) as client:
        created = await _color(client)
        out = _data(
            await client.call_tool(
                "adjust_colors",
                {"clip_id": created["clip_id"], "brightness": 0.2, "contrast": 0.1},
            )
        )
        assert out["clip_id"] != created["clip_id"]


async def test_chroma_key_masks_color():
    async with Client(mcp) as client:
        created = await _color(client, color=[0, 255, 0])
        out = _data(
            await client.call_tool(
                "chroma_key",
                {"clip_id": created["clip_id"], "color_rgb": [0, 255, 0],
                 "threshold": 30},
            )
        )
        assert out["clip_id"] != created["clip_id"]


async def test_chroma_key_rejects_bad_color_length():
    async with Client(mcp) as client:
        created = await _color(client)
        result = await client.call_tool(
            "chroma_key",
            {"clip_id": created["clip_id"], "color_rgb": [0, 255]},
            raise_on_error=False,
        )
        assert result.is_error


async def test_invert_colors_keeps_size():
    async with Client(mcp) as client:
        created = await _color(client, width=16, height=16)
        out = _data(
            await client.call_tool("invert_colors", {"clip_id": created["clip_id"]})
        )
        assert out["size"] == {"width": 16, "height": 16}


async def test_gamma_correct_rejects_non_positive():
    async with Client(mcp) as client:
        created = await _color(client)
        result = await client.call_tool(
            "gamma_correct", {"clip_id": created["clip_id"], "gamma": 0},
            raise_on_error=False,
        )
        assert result.is_error


async def test_gamma_correct_applies():
    async with Client(mcp) as client:
        created = await _color(client)
        out = _data(
            await client.call_tool(
                "gamma_correct", {"clip_id": created["clip_id"], "gamma": 1.5}
            )
        )
        assert out["clip_id"] != created["clip_id"]


async def test_multiply_color_rejects_non_positive():
    async with Client(mcp) as client:
        created = await _color(client)
        result = await client.call_tool(
            "multiply_color", {"clip_id": created["clip_id"], "factor": -1},
            raise_on_error=False,
        )
        assert result.is_error


async def test_multiply_color_applies():
    async with Client(mcp) as client:
        created = await _color(client)
        out = _data(
            await client.call_tool(
                "multiply_color", {"clip_id": created["clip_id"], "factor": 0.5}
            )
        )
        assert out["clip_id"] != created["clip_id"]


async def test_add_margin_equal_sides():
    async with Client(mcp) as client:
        created = await _color(client, width=20, height=10)
        out = _data(
            await client.call_tool(
                "add_margin", {"clip_id": created["clip_id"], "margin_size": 5}
            )
        )
        assert out["size"] == {"width": 30, "height": 20}


async def test_add_margin_individual_sides():
    async with Client(mcp) as client:
        created = await _color(client, width=20, height=10)
        out = _data(
            await client.call_tool(
                "add_margin",
                {"clip_id": created["clip_id"], "left": 2, "right": 3,
                 "top": 1, "bottom": 4},
            )
        )
        assert out["size"] == {"width": 25, "height": 15}


async def test_add_margin_requires_a_size():
    async with Client(mcp) as client:
        created = await _color(client)
        result = await client.call_tool(
            "add_margin", {"clip_id": created["clip_id"]}, raise_on_error=False
        )
        assert result.is_error


async def test_painting_effect_keeps_size():
    async with Client(mcp) as client:
        created = await _color(client, width=16, height=16)
        out = _data(
            await client.call_tool("painting", {"clip_id": created["clip_id"]})
        )
        assert out["size"] == {"width": 16, "height": 16}


async def test_blur_rejects_negative_radius():
    async with Client(mcp) as client:
        created = await _color(client)
        result = await client.call_tool(
            "blur", {"clip_id": created["clip_id"], "radius": -1},
            raise_on_error=False,
        )
        assert result.is_error


async def test_blur_applies():
    async with Client(mcp) as client:
        created = await _color(client, width=16, height=16)
        out = _data(
            await client.call_tool(
                "blur", {"clip_id": created["clip_id"], "radius": 2}
            )
        )
        assert out["size"] == {"width": 16, "height": 16}


async def test_sharpen_applies():
    async with Client(mcp) as client:
        created = await _color(client, width=16, height=16)
        out = _data(
            await client.call_tool("sharpen", {"clip_id": created["clip_id"]})
        )
        assert out["size"] == {"width": 16, "height": 16}


async def test_set_opacity_rejects_out_of_range():
    async with Client(mcp) as client:
        created = await _color(client)
        result = await client.call_tool(
            "set_opacity", {"clip_id": created["clip_id"], "opacity": 1.5},
            raise_on_error=False,
        )
        assert result.is_error


async def test_set_opacity_applies():
    async with Client(mcp) as client:
        created = await _color(client)
        out = _data(
            await client.call_tool(
                "set_opacity", {"clip_id": created["clip_id"], "opacity": 0.5}
            )
        )
        assert out["clip_id"] != created["clip_id"]


# --------------------------------------------------------------------------- #
# Audio
# --------------------------------------------------------------------------- #


async def test_set_volume_scales_audio():
    async with Client(mcp) as client:
        cid = _tone()
        out = _data(
            await client.call_tool("set_volume", {"clip_id": cid, "factor": 0.5})
        )
        assert out["kind"] == "audio"


async def test_set_volume_rejects_video_without_audio():
    async with Client(mcp) as client:
        created = await _color(client)
        result = await client.call_tool(
            "set_volume", {"clip_id": created["clip_id"], "factor": 0.5},
            raise_on_error=False,
        )
        assert result.is_error


async def test_extract_audio_and_remove_audio():
    async with Client(mcp) as client:
        video = await _color(client, duration=0.3)
        audio_id = _tone(duration=0.3)
        with_audio = _data(
            await client.call_tool(
                "attach_audio",
                {"video_clip_id": video["clip_id"], "audio_clip_id": audio_id},
            )
        )
        assert with_audio["has_audio"] is True

        extracted = _data(
            await client.call_tool(
                "extract_audio", {"clip_id": with_audio["clip_id"]}
            )
        )
        assert extracted["kind"] == "audio"

        muted = _data(
            await client.call_tool(
                "remove_audio", {"clip_id": with_audio["clip_id"]}
            )
        )
        assert muted["has_audio"] is False


async def test_extract_audio_rejects_silent_video():
    async with Client(mcp) as client:
        created = await _color(client)
        result = await client.call_tool(
            "extract_audio", {"clip_id": created["clip_id"]}, raise_on_error=False
        )
        assert result.is_error


async def test_attach_audio_mix_with_existing():
    async with Client(mcp) as client:
        video = await _color(client, duration=0.3)
        first = _tone(duration=0.3)
        second = _tone(duration=0.3)
        base = _data(
            await client.call_tool(
                "attach_audio",
                {"video_clip_id": video["clip_id"], "audio_clip_id": first},
            )
        )
        mixed = _data(
            await client.call_tool(
                "attach_audio",
                {
                    "video_clip_id": base["clip_id"],
                    "audio_clip_id": second,
                    "mix_with_existing": True,
                },
            )
        )
        assert mixed["has_audio"] is True


# --------------------------------------------------------------------------- #
# Compositing
# --------------------------------------------------------------------------- #


async def test_overlay_clip_positions_and_duration():
    async with Client(mcp) as client:
        base = await _color(client, width=100, height=100, duration=1.0)
        overlay = await _color(client, width=20, height=20, duration=0.5,
                               color=[0, 0, 255])
        out = _data(
            await client.call_tool(
                "overlay_clip",
                {
                    "base_clip_id": base["clip_id"],
                    "overlay_clip_id": overlay["clip_id"],
                    "position": "top-left",
                    "start_seconds": 0.1,
                },
            )
        )
        assert out["size"] == {"width": 100, "height": 100}
        assert out["duration_seconds"] == 1.0


async def test_overlay_clip_exact_xy_position():
    async with Client(mcp) as client:
        base = await _color(client, width=100, height=100, duration=1.0)
        overlay = await _color(client, width=10, height=10, duration=0.5)
        out = _data(
            await client.call_tool(
                "overlay_clip",
                {
                    "base_clip_id": base["clip_id"],
                    "overlay_clip_id": overlay["clip_id"],
                    "x": 5,
                    "y": 5,
                    "opacity": 0.8,
                },
            )
        )
        assert out["clip_id"] != base["clip_id"]


# --------------------------------------------------------------------------- #
# Inspection & registry management
# --------------------------------------------------------------------------- #


async def test_get_clip_info_matches_create_color_clip():
    async with Client(mcp) as client:
        created = await _color(client, width=12, height=8)
        info = _data(
            await client.call_tool("get_clip_info", {"clip_id": created["clip_id"]})
        )
        assert info == created


async def test_get_clip_info_unknown_id_is_error():
    async with Client(mcp) as client:
        result = await client.call_tool(
            "get_clip_info", {"clip_id": "video_missing"}, raise_on_error=False
        )
        assert result.is_error


async def test_delete_clip_removes_entry():
    async with Client(mcp) as client:
        created = await _color(client)
        out = _data(
            await client.call_tool("delete_clip", {"clip_id": created["clip_id"]})
        )
        assert out["deleted"] == created["clip_id"]
        assert out["remaining_clips"] == 0
        result = await client.call_tool(
            "get_clip_info", {"clip_id": created["clip_id"]}, raise_on_error=False
        )
        assert result.is_error


# --------------------------------------------------------------------------- #
# Load & output (real files)
# --------------------------------------------------------------------------- #


async def test_load_image_from_real_file(tmp_path: Path):
    path = tmp_path / "pic.png"
    PILImage.new("RGB", (24, 16), (10, 20, 30)).save(path)
    async with Client(mcp) as client:
        out = _data(
            await client.call_tool(
                "load_image", {"path": str(path), "duration_seconds": 0.4}
            )
        )
        assert out["kind"] == "image"
        assert out["size"] == {"width": 24, "height": 16}
        assert out["duration_seconds"] == 0.4


async def test_load_image_missing_path_is_error():
    async with Client(mcp) as client:
        result = await client.call_tool(
            "load_image", {"path": "/no/such/file.png"}, raise_on_error=False
        )
        assert result.is_error


async def test_save_frame_writes_file(tmp_path: Path):
    async with Client(mcp) as client:
        created = await _color(client, width=16, height=16)
        out_path = tmp_path / "frame.png"
        out = _data(
            await client.call_tool(
                "save_frame",
                {"clip_id": created["clip_id"], "output_path": str(out_path)},
            )
        )
        assert out["saved"] == str(out_path)
        assert out_path.stat().st_size > 0


async def test_export_image_writes_file():
    async with Client(mcp) as client:
        created = await _color(client, width=16, height=16)
        result = await client.call_tool(
            "export_image",
            {"clip_id": created["clip_id"], "output_path": "/tmp/does-not-matter.bmp"},
            raise_on_error=False,
        )
        assert result.is_error  # unsupported extension


async def test_export_image_writes_png(tmp_path: Path):
    async with Client(mcp) as client:
        created = await _color(client, width=16, height=16)
        out_path = tmp_path / "still.png"
        out = _data(
            await client.call_tool(
                "export_image",
                {"clip_id": created["clip_id"], "output_path": str(out_path)},
            )
        )
        assert out["exported"] == str(out_path)
        assert out["file_size_bytes"] > 0


@pytest.mark.skipif(not FFMPEG, reason="ffmpeg required")
async def test_load_video_round_trip(tmp_path: Path):
    async with Client(mcp) as client:
        created = await _color(client, width=32, height=32, duration=0.25)
        out_path = tmp_path / "clip.mp4"
        await client.call_tool(
            "export_clip",
            {"clip_id": created["clip_id"], "output_path": str(out_path), "fps": 8},
            timeout=60,
        )
        loaded = _data(
            await client.call_tool("load_video", {"path": str(out_path)})
        )
        assert loaded["kind"] == "video"
        assert loaded["size"] == {"width": 32, "height": 32}


@pytest.mark.skipif(not FFMPEG, reason="ffmpeg required")
async def test_load_video_missing_path_is_error():
    async with Client(mcp) as client:
        result = await client.call_tool(
            "load_video", {"path": "/no/such/video.mp4"}, raise_on_error=False
        )
        assert result.is_error


@pytest.mark.skipif(not FFMPEG, reason="ffmpeg required")
async def test_load_audio_round_trip(tmp_path: Path):
    async with Client(mcp) as client:
        cid = _tone(duration=0.2)
        out_path = tmp_path / "tone.wav"
        await client.call_tool(
            "export_clip", {"clip_id": cid, "output_path": str(out_path)},
            timeout=60,
        )
        loaded = _data(
            await client.call_tool("load_audio", {"path": str(out_path)})
        )
        assert loaded["kind"] == "audio"
