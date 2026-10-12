"""Workflow prompts. They tell the model which tools to call, in order."""

from __future__ import annotations

from moviepy_mcp.server import mcp


@mcp.prompt(tags={"core"})
def short_form(source_path: str, title: str = "") -> str:
    """Turn a video into a 9:16 short with a burned-in title."""
    title_step = (
        f" Create a text clip with the title {title!r} and overlay it near the top "
        "for the first 3 seconds."
        if title else ""
    )
    return (
        "Make a vertical short from a file in the media workspace. "
        f"Load {source_path} with load_video. "
        "Trim to the useful range. reframe to 1080x1920 with mode crop. "
        f"{title_step} preview_frame at the start and the midpoint. "
        "export_clip with preset reels_1080x1920. "
        "Use the newest clip_id at every step."
    )


@mcp.prompt(tags={"core"})
def podcast_cut(source_path: str) -> str:
    """Jump-cut silence out of a podcast or vlog and export AAC."""
    return (
        f"Load {source_path}. If it is video, extract_audio when you only need the sound. "
        "remove_silence with the default threshold, then preview is not available for audio. "
        "export_clip with preset podcast_aac. "
        "If the music should sit under speech, duck_audio the music clip using the voice clip."
    )


@mcp.prompt(tags={"core"})
def slideshow(image_directory: str, audio_path: str = "") -> str:
    """Build a slideshow from images in one folder."""
    audio = (
        f" load_audio {audio_path} and pass that clip_id as audio_clip_id."
        if audio_path else ""
    )
    return (
        f"Create a slideshow from the images in {image_directory}. "
        "Pass those paths to create_slideshow, about 3 seconds each, "
        f"with a short crossfade.{audio} "
        "preview_frame the first slide, then export_clip with preset youtube_1080p."
    )


@mcp.prompt(tags={"core"})
def titled_intro(source_path: str, title: str, start_seconds: float = 0,
                 end_seconds: float = 0) -> str:
    """Trim a clip, put a title on it, fade out, and export."""
    window = ""
    if end_seconds > start_seconds:
        window = f" Trim from {start_seconds} to {end_seconds} seconds."
    return (
        f"Load {source_path}.{window} "
        f"create_text_clip for {title!r}. If text rendering fails, list_fonts and "
        "pass one of those paths as font. overlay_clip the title on top. "
        "fade the end. preview_frame, then export_clip."
    )
