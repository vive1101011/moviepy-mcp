# MoviePy MCP Server

An LLM-friendly [Model Context Protocol](https://modelcontextprotocol.io) server for video and audio editing, built on **MoviePy v2** and **FastMCP**. It lets MCP clients (Claude Desktop, Claude Code, Cursor, etc.) trim, merge, resize, overlay, mix audio, and render video through natural language.

## Design

The server is built around an **in-memory clip registry**:

1. `load_video` / `load_audio` / `load_image` returns a `clip_id`.
2. Every editing tool takes a `clip_id` and returns a **new** `clip_id` — operations chain without touching disk, and source clips are never mutated, so an agent can branch or retry any step.
3. `export_clip` is the only tool that renders a file.

Every tool returns uniform metadata (`duration_seconds`, `size`, `fps`, `has_audio`, `history`), so the model always knows the state of an edit without extra probing. Errors are actionable: they name the bad argument, list known clip_ids, and suggest the fix.

## Requirements

- Python 3.10+
- FFmpeg on your PATH (`sudo apt install ffmpeg` / `brew install ffmpeg`)

## Installation

```bash
git clone <your-repo-url> moviepy-mcp
cd moviepy-mcp
pip install -e .
# or with uv:
uv pip install -e .
```

Run it directly to verify:

```bash
moviepy-mcp
```

## Client configuration

### Cursor (`~/.cursor/mcp.json`)

Add a `moviepy` entry under `mcpServers` (same shape as other local `uv` servers). Restart Cursor or reload MCP after saving:

```json
{
  "mcpServers": {
    "moviepy": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "C:/Users/vivek/projects/moviepy-mcp",
        "python",
        "src/moviepy_mcp/server.py"
      ]
    }
  }
}
```

Replace the `--directory` path with your clone of this repo. `uv run` uses the project’s `pyproject.toml` / `.venv`, so you do not need a global `moviepy-mcp` install.

### Claude Desktop / Claude Code (stdio)

If the package is installed (`pip install -e .` or `uv pip install -e .`):

```json
{
  "mcpServers": {
    "moviepy": {
      "command": "moviepy-mcp"
    }
  }
}
```

Or without installing, via uv (one-shot deps):

```json
{
  "mcpServers": {
    "moviepy": {
      "command": "uv",
      "args": [
        "run",
        "--with",
        "fastmcp",
        "--with",
        "moviepy",
        "python",
        "/path/to/moviepy-mcp/src/moviepy_mcp/server.py"
      ]
    }
  }
}
```

## Tools

| Category | Tools |
|---|---|
| Load / create | `load_video`, `load_audio`, `load_image`, `create_color_clip`, `create_text_clip` |
| Inspect | `list_clips`, `get_clip_info`, `delete_clip` |
| Time | `trim`, `concatenate`, `change_speed`, `loop_clip` |
| Geometry | `resize`, `crop`, `rotate`, `mirror` |
| Visual FX | `fade`, `to_grayscale`, `adjust_colors`, `chroma_key` |
| Audio | `set_volume`, `extract_audio`, `remove_audio`, `attach_audio` |
| Compositing | `overlay_clip` |
| Output | `save_frame`, `export_clip` |

## Example agent workflow

> "Take intro.mp4, keep seconds 5–20, add the title 'Q2 Review' for the first 4 seconds, fade out at the end, and save as final.mp4."

The agent will typically call:

```
load_video(path="intro.mp4")                          -> video_a1b2c3d4
trim(clip_id="video_a1b2c3d4", start_seconds=5, end_seconds=20)
create_text_clip(text="Q2 Review", font_size=64, duration_seconds=4)
overlay_clip(base_clip_id=..., overlay_clip_id=..., position="top")
fade(clip_id=..., fade_out_seconds=1.5)
export_clip(clip_id=..., output_path="final.mp4")
```

## Notes

- Text rendering requires a font; pass `font="/path/to/font.ttf"` to `create_text_clip` if the default is unavailable on your system.
- GIF export uses `write_gif` (defaults to 12 fps); everything else goes through FFmpeg.
- The registry lives in server memory — restarting the server clears all clip_ids.

## License

MIT
