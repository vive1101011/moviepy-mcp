"""Workspace boundary, registry caps, presets, and the smaller tool contract."""

import os
from pathlib import Path

import pytest
from fastmcp import Client
from PIL import Image as PILImage

from moviepy_mcp.paths import resolve_read, resolve_write
from moviepy_mcp.presets import get_preset
from moviepy_mcp.server import _register, mcp


def test_resolve_write_allows_temp(tmp_path: Path):
    target = tmp_path / "nested" / "out.mp4"
    assert resolve_write(str(target)) == str(target)
    assert target.parent.is_dir()


def test_resolve_read_rejects_files_outside_the_workspace(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MOVIEPY_MCP_MEDIA_ROOT", str(tmp_path))
    windir = os.environ.get("WINDIR")
    outside = Path(windir) / "win.ini" if windir else Path("/etc/hosts")
    if not outside.is_file():
        outside = Path("/etc/hosts")
    if not outside.is_file():
        pytest.skip("no file outside the workspace and temp dir")
    with pytest.raises(ValueError, match="media workspace"):
        resolve_read(str(outside))


def test_preset_names():
    spec = get_preset("reels_1080x1920")
    assert (spec.width, spec.height, spec.loudness_lufs) == (1080, 1920, -14)
    with pytest.raises(ValueError, match="Unknown preset"):
        get_preset("cinema")


def test_registry_drops_the_oldest_unused_clip(monkeypatch):
    monkeypatch.setenv("MOVIEPY_MCP_MAX_CLIPS", "2")
    from moviepy_mcp.registry import current_ids

    class Dummy:
        w = 8
        h = 8
        duration = 1

        def close(self):
            self.closed = True

    first = _register(Dummy(), "video", "a", None, "create_color_clip")
    _register(Dummy(), "video", "b", None, "create_color_clip")
    third = _register(Dummy(), "video", "c", None, "create_color_clip")
    ids = current_ids()
    assert first not in ids
    assert third in ids
    assert len(ids) == 2


def test_project_reopens_a_loaded_image(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MOVIEPY_MCP_PERSIST", "1")
    monkeypatch.setenv("MOVIEPY_MCP_MEDIA_ROOT", str(tmp_path))
    image = tmp_path / "still.png"
    PILImage.new("RGB", (8, 8), (255, 0, 0)).save(image)

    from moviepy import ImageClip

    from moviepy_mcp import registry

    clip = ImageClip(str(image)).with_duration(1)
    clip_id = registry._register(
        clip, "image", "still", None, "load_image(still.png)",
        source_path=str(image), args={"duration_seconds": 1.0},
    )
    for bucket in list(registry._SESSIONS.values()):
        for entry in bucket.values():
            if entry.clip is not clip:
                entry.clip.close()
        bucket.clear()
    registry._SESSIONS.clear()
    registry._RESTORED.clear()
    restored = registry._get(clip_id)
    assert restored.kind == "image"
    assert restored.source_path == str(image)


def test_sessions_are_separate_buckets():
    from moviepy_mcp.registry import bucket_for

    assert bucket_for("alpha") is not bucket_for("beta")


@pytest.mark.asyncio
async def test_browser_cookies_stay_off():
    async with Client(mcp) as client:
        result = await client.call_tool(
            "download_video",
            {
                "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                "cookies_from_browser": "chrome",
            },
            raise_on_error=False,
        )
    assert result.is_error


@pytest.mark.asyncio
async def test_unknown_preset_is_an_error():
    async with Client(mcp) as client:
        created = await client.call_tool(
            "create_color_clip",
            {"width": 16, "height": 16, "color_rgb": [0, 0, 0], "duration_seconds": 0.2},
        )
        data = created.data if isinstance(created.data, dict) else created.data
        clip_id = data["clip_id"] if isinstance(data, dict) else data.clip_id
        result = await client.call_tool(
            "export_clip",
            {"clip_id": clip_id, "output_path": "nope.mp4", "preset": "cinema"},
            raise_on_error=False,
        )
    assert result.is_error


@pytest.mark.asyncio
async def test_prompts_and_advanced_tag_are_visible():
    async with Client(mcp) as client:
        prompts = {item.name for item in await client.list_prompts()}
        tools = {tool.name: tool for tool in await client.list_tools()}
    assert {"short_form", "podcast_cut", "slideshow", "titled_intro"} <= prompts
    assert "duck_audio" in tools
    assert "list_fonts" in tools
    meta = tools["blur"].meta or {}
    tags = (meta.get("fastmcp") or meta.get("_fastmcp") or {}).get("tags") or []
    assert "advanced" in tags


@pytest.mark.asyncio
async def test_duck_audio_keeps_duration():
    import numpy as np
    from moviepy import AudioClip

    def tone(freq: float):
        clip = AudioClip(
            lambda t: 0.2 * np.sin(2 * np.pi * freq * t),
            duration=0.2,
            fps=8000,
        )
        return _register(clip, "audio", f"tone-{freq}", None, "test")

    music = tone(220)
    voice = tone(440)
    async with Client(mcp) as client:
        result = await client.call_tool(
            "duck_audio",
            {"music_clip_id": music, "voice_clip_id": voice, "duck_factor": 0.25},
        )
    data = result.data if isinstance(result.data, dict) else result.data
    payload = data if isinstance(data, dict) else data.model_dump()
    assert payload["kind"] == "audio"
    assert payload["duration_seconds"] == 0.2


def test_playback_payload_embeds_small_files(tmp_path: Path):
    from moviepy_mcp.tools.output import playback_payload

    path = tmp_path / "frame.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n")
    small = playback_payload(str(path), limit=1000)
    assert small["ready"] is True
    assert small["kind"] == "image"
    assert small["src"].startswith("data:image/png;base64,")
    large = playback_payload(str(path), limit=4)
    assert large["ready"] is False
    assert large["src"] == ""
    assert large["path"] == str(path)
    from moviepy_mcp.tools.output import _download_card

    card = _download_card(str(path), size=path.stat().st_size, kind="image")
    assert card.to_json()["view"]


async def test_chat_upload_and_download_tools():
    async with Client(mcp) as client:
        listed = await client.list_tools()
    tools = {tool.name: tool for tool in listed}
    assert "clip_id" in (tools["file_manager"].description or "")
    assert "player" in (tools["download_media"].description or "")
    assert "ctx" not in tools["download_media"].inputSchema.get("properties", {})
    assert "load_playback" not in tools
    async with Client(mcp) as client:
        opened = await client.call_tool("file_manager", {})
    assert opened.data is not None
