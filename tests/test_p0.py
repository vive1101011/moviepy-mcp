import base64
import io
import json
import shutil
from pathlib import Path

import pytest
from fastmcp import Client
from PIL import Image as PILImage

from moviepy_mcp.server import _REGISTRY, _register, _ytdlp_progress_hook, mcp

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _tool_data(result) -> dict:
    if isinstance(result.data, dict):
        return result.data
    for block in result.content:
        text = getattr(block, "text", None)
        if text:
            return json.loads(text)
    raise AssertionError(f"no dict payload in {result.content!r}")


def _png_bytes(result) -> bytes:
    for block in result.content:
        data = getattr(block, "data", None)
        typ = getattr(block, "type", None)
        mime = getattr(block, "mimeType", None) or getattr(block, "mime_type", None)
        if data is None:
            continue
        if typ == "image" or (isinstance(mime, str) and mime.startswith("image/")):
            raw = data if isinstance(data, bytes) else base64.b64decode(data)
            assert raw.startswith(PNG_MAGIC)
            return raw
    raise AssertionError(f"no image block in {result.content!r}")


def _resource_bytes(contents) -> bytes:
    item = contents[0]
    blob = getattr(item, "blob", None)
    if blob is not None:
        return blob if isinstance(blob, bytes) else base64.b64decode(blob)
    text = getattr(item, "text", None)
    if text is not None:
        return text.encode("utf-8") if isinstance(text, str) else text
    raise AssertionError(f"no blob/text in {contents!r}")


async def _color_clip(client, width=32, height=32, duration=1.0, color=None):
    result = await client.call_tool(
        "create_color_clip",
        {
            "width": width,
            "height": height,
            "color_rgb": color or [255, 0, 0],
            "duration_seconds": duration,
        },
    )
    return _tool_data(result)


@pytest.mark.asyncio
async def test_preview_frame_returns_png_and_metadata():
    async with Client(mcp) as client:
        created = await _color_clip(client, duration=1.0)
        result = await client.call_tool(
            "preview_frame",
            {"clip_id": created["clip_id"], "time_seconds": 0.0, "max_width": 0},
        )
        png = _png_bytes(result)
        img = PILImage.open(io.BytesIO(png))
        assert img.size == (32, 32)
        meta = _tool_data(result)
        assert meta["clip_id"] == created["clip_id"]
        assert meta["preview_size"] == {"width": 32, "height": 32}
        assert meta["time_seconds"] == 0.0


@pytest.mark.asyncio
async def test_preview_frame_downscales():
    async with Client(mcp) as client:
        created = await _color_clip(client, width=800, height=200)
        result = await client.call_tool(
            "preview_frame",
            {"clip_id": created["clip_id"], "max_width": 200},
        )
        img = PILImage.open(io.BytesIO(_png_bytes(result)))
        assert img.size == (200, 50)
        assert _tool_data(result)["preview_size"] == {"width": 200, "height": 50}


@pytest.mark.asyncio
async def test_preview_frame_unknown_id_is_error():
    async with Client(mcp) as client:
        result = await client.call_tool(
            "preview_frame",
            {"clip_id": "video_doesnotexist"},
            raise_on_error=False,
        )
        assert result.is_error


@pytest.mark.asyncio
async def test_preview_frame_rejects_audio():
    from moviepy import AudioClip

    clip = AudioClip(lambda t: 0 * t, duration=0.1, fps=44100)
    clip_id = _register(clip, "audio", "silence", None, "test")
    async with Client(mcp) as client:
        result = await client.call_tool(
            "preview_frame",
            {"clip_id": clip_id},
            raise_on_error=False,
        )
        assert result.is_error
        assert clip_id in _REGISTRY


@pytest.mark.asyncio
async def test_preview_frame_rejects_time_past_duration():
    async with Client(mcp) as client:
        created = await _color_clip(client, duration=1.0)
        result = await client.call_tool(
            "preview_frame",
            {"clip_id": created["clip_id"], "time_seconds": 5.0},
            raise_on_error=False,
        )
        assert result.is_error


@pytest.mark.asyncio
async def test_clip_info_and_frame_resources():
    async with Client(mcp) as client:
        created = await _color_clip(client, width=64, height=32)
        clip_id = created["clip_id"]
        info = await client.read_resource(f"clip://{clip_id}/info")
        payload = json.loads(info[0].text)
        assert payload["clip_id"] == clip_id
        assert payload["size"] == {"width": 64, "height": 32}

        frame = await client.read_resource(f"clip://{clip_id}/frame")
        png = _resource_bytes(frame)
        assert png.startswith(PNG_MAGIC)
        img = PILImage.open(io.BytesIO(png))
        assert img.size == (64, 32)


@pytest.mark.asyncio
async def test_trim_returns_new_id_and_duration():
    async with Client(mcp) as client:
        created = await _color_clip(client, duration=2.0)
        trimmed = _tool_data(
            await client.call_tool(
                "trim",
                {
                    "clip_id": created["clip_id"],
                    "start_seconds": 0.5,
                    "end_seconds": 1.5,
                },
            )
        )
        assert trimmed["clip_id"] != created["clip_id"]
        assert trimmed["duration_seconds"] == 1.0
        assert trimmed["history"][0].startswith("create_color_clip")
        assert any("trim" in step for step in trimmed["history"])
        listed = _tool_data(await client.call_tool("list_clips", {}))
        assert listed["count"] == 2


@pytest.mark.asyncio
async def test_export_and_download_omit_ctx_from_schema():
    async with Client(mcp) as client:
        tools = {t.name: t for t in await client.list_tools()}
        for name in ("export_clip", "download_video"):
            props = tools[name].inputSchema["properties"]
            assert "ctx" not in props
            assert "clip_id" in props or "url" in props


@pytest.mark.asyncio
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
async def test_export_clip_reports_progress(tmp_path: Path):
    events: list[tuple[float, float | None, str | None]] = []

    async def handler(progress: float, total: float | None, message: str | None) -> None:
        events.append((progress, total, message))

    async with Client(mcp) as client:
        created = await _color_clip(client, width=32, height=32, duration=0.25)
        out = str(tmp_path / "out.mp4")
        result = await client.call_tool(
            "export_clip",
            {"clip_id": created["clip_id"], "output_path": out, "fps": 8},
            progress_handler=handler,
            timeout=60,
        )
        data = _tool_data(result)
        assert data["exported"] == out
        assert Path(out).stat().st_size > 0
        assert data["file_size_bytes"] == Path(out).stat().st_size

    assert events, "expected progress notifications during export"
    assert any(msg == "starting export" for _, _, msg in events)
    assert any(msg == "export complete" for _, _, msg in events)


def test_ytdlp_progress_hook_maps_download_status():
    seen: list[tuple[float, float | None, str]] = []
    hook = _ytdlp_progress_hook(lambda p, t, m: seen.append((p, t, m)))
    hook({"status": "downloading", "downloaded_bytes": 25, "total_bytes": 100})
    hook({"status": "downloading", "downloaded_bytes": 50, "total_bytes_estimate": 200})
    hook({"status": "finished", "total_bytes": 100})
    hook({"status": "error"})
    assert seen == [
        (25.0, 100.0, "downloading"),
        (50.0, 200.0, "downloading"),
        (100.0, 100.0, "finished"),
    ]
