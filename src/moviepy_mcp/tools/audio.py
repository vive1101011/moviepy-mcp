"""Volume, mix, silence, and ducking."""

from __future__ import annotations

import asyncio
import os
import uuid
from typing import Any, Optional

from moviepy import CompositeAudioClip, afx, concatenate_audioclips, concatenate_videoclips
from mcp.types import ToolAnnotations

from moviepy_mcp.paths import resolve_dir, resolve_read, resolve_write, workspace_dir
from moviepy_mcp.registry import ClipEntry, _describe, _get, _register
from moviepy_mcp.runtime import LOCK, serialized
from moviepy_mcp.server import mcp
from moviepy_mcp.editing import _detect_silence

READ = ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False)
EDIT = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False)
DESTROY = ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=False)
NET = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=True)


def _apply_audio_effect(entry: ClipEntry, effect: Any, op: str) -> dict:
    if entry.kind == "audio":
        new = entry.clip.with_effects([effect])
        kind = "audio"
    else:
        if entry.clip.audio is None:
            raise ValueError(
                f"clip_id has no audio track. Load audio or attach_audio first.")
        new = entry.clip.with_audio(entry.clip.audio.with_effects([effect]))
        kind = "video" if entry.kind == "image" else entry.kind
    nid = _register(new, kind, entry.label, entry, op)
    return _describe(nid)

@mcp.tool(tags={'core'}, annotations=EDIT)
@serialized
def set_volume(clip_id: str, factor: float) -> dict:
    """Scale volume: 0.5 = half, 2.0 = double, 0 = silent.

    Works on audio clips and on the audio track of video clips.
    """
    entry = _get(clip_id)
    if entry.kind == "video" and entry.clip.audio is None:
        raise ValueError("This video has no audio track.")
    new = entry.clip.with_volume_scaled(factor)
    nid = _register(new, entry.kind, entry.label, entry, f"volume x{factor}")
    return _describe(nid)

@mcp.tool(tags={'advanced'}, annotations=EDIT)
@serialized
def normalize_audio(clip_id: str) -> dict:
    """Normalize volume to 0 dB. Works on audio clips and video soundtracks."""
    entry = _get(clip_id)
    return _apply_audio_effect(entry, afx.AudioNormalize(), "normalize_audio")

@mcp.tool(tags={'advanced'}, annotations=EDIT)
@serialized
def delay_audio(clip_id: str, offset_seconds: float = 0.2,
                n_repeats: int = 8, decay: float = 1.0) -> dict:
    """Echo: repeat the audio n_repeats times, offset_seconds apart.

    decay < 1 fades repeats. Works on audio clips and video soundtracks.
    """
    if offset_seconds <= 0:
        raise ValueError("offset_seconds must be > 0.")
    if n_repeats < 1:
        raise ValueError("n_repeats must be >= 1.")
    entry = _get(clip_id)
    return _apply_audio_effect(
        entry,
        afx.AudioDelay(offset=offset_seconds, n_repeats=n_repeats, decay=decay),
        f"delay_audio({offset_seconds}x{n_repeats})",
    )

@mcp.tool(tags={'advanced'}, annotations=EDIT)
@serialized
def set_stereo_volume(clip_id: str, left: float = 1.0,
                      right: float = 1.0) -> dict:
    """Set left/right channel gain (pan). Needs a stereo track.

    1.0 = unchanged, 0 = mute that side. Works on audio or video-with-audio.
    """
    if left < 0 or right < 0:
        raise ValueError("left and right must be >= 0.")
    entry = _get(clip_id)
    try:
        return _apply_audio_effect(
            entry, afx.MultiplyStereoVolume(left=left, right=right),
            f"stereo_volume(L={left}, R={right})")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            f"set_stereo_volume needs a stereo audio track: {exc}"
        ) from exc

@mcp.tool(tags={'core'}, annotations=EDIT)
@serialized
def remove_silence(clip_id: str, silence_threshold_db: float = -40.0,
                   min_silence_seconds: float = 0.5,
                   padding_seconds: float = 0.1) -> dict:
    """Cut silent gaps out of a clip (podcast/vlog jump-cut editing).

    Analyzes the clip's audio track, finds runs of near-silence at least
    min_silence_seconds long, and removes them — video and audio stay in
    sync when clip_id is a video.

    Args:
        clip_id: Audio clip, or video clip with an audio track.
        silence_threshold_db: Loudness below this (dBFS) counts as silence.
            Typical range: -50 (very sensitive) to -30 (only near-total
            silence).
        min_silence_seconds: Minimum gap length to remove.
        padding_seconds: Seconds of the silent gap kept on each side of a
            cut, so words are not clipped.
    """
    if min_silence_seconds <= 0:
        raise ValueError("min_silence_seconds must be > 0.")
    if padding_seconds < 0:
        raise ValueError("padding_seconds must be >= 0.")
    entry = _get(clip_id)
    audio = entry.clip if entry.kind == "audio" else entry.clip.audio
    if audio is None:
        raise ValueError("clip_id has no audio track to analyze.")
    duration = entry.clip.duration or 0.0
    if not duration:
        raise ValueError("Clip has no duration.")

    silences = _detect_silence(audio, silence_threshold_db, min_silence_seconds)
    if not silences:
        raise ValueError(
            "No silence found at or above min_silence_seconds with this "
            "threshold. Try a higher silence_threshold_db (e.g. -30) or a "
            "shorter min_silence_seconds.")

    keep: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in silences:
        cut_start = min(duration, start + padding_seconds)
        cut_end = max(0.0, end - padding_seconds)
        if cut_start < cut_end:
            if cut_start > cursor:
                keep.append((cursor, cut_start))
            cursor = cut_end
    if cursor < duration:
        keep.append((cursor, duration))
    keep = [(s, e) for s, e in keep if e - s > 1e-3]
    if not keep:
        raise ValueError(
            "Removing silence would leave nothing; lower min_silence_seconds "
            "or padding_seconds.")

    segments = [entry.clip.subclipped(s, e) for s, e in keep]
    if len(segments) == 1:
        new = segments[0]
    elif entry.kind == "audio":
        new = concatenate_audioclips(segments)
    else:
        new = concatenate_videoclips(segments, method="compose")
    nid = _register(new, entry.kind, entry.label, entry,
                    f"remove_silence({len(silences)} gaps)")
    return _describe(nid)

@mcp.tool(tags={'core'}, annotations=EDIT)
@serialized
def extract_audio(clip_id: str) -> dict:
    """Pull the audio track out of a video as a new audio clip."""
    entry = _get(clip_id, expect=("video",))
    if entry.clip.audio is None:
        raise ValueError("This video has no audio track to extract.")
    nid = _register(entry.clip.audio, "audio", f"audio of {entry.label}",
                    entry, "extract_audio")
    return _describe(nid)

@mcp.tool(tags={'core'}, annotations=EDIT)
@serialized
def remove_audio(clip_id: str) -> dict:
    """Return a muted copy of a video clip (audio track removed)."""
    entry = _get(clip_id, expect=("video",))
    new = entry.clip.without_audio()
    nid = _register(new, "video", entry.label, entry, "remove_audio")
    return _describe(nid)

@mcp.tool(tags={'core'}, annotations=EDIT)
@serialized
def attach_audio(video_clip_id: str, audio_clip_id: str,
                 mix_with_existing: bool = False) -> dict:
    """Set or mix an audio track onto a video clip.

    Args:
        video_clip_id: The video to receive audio.
        audio_clip_id: The audio clip to attach.
        mix_with_existing: If True and the video already has audio, mix both;
            if False, replace the existing track.
    """
    ventry = _get(video_clip_id, expect=("video", "image"))
    aentry = _get(audio_clip_id, expect=("audio",))
    audio = aentry.clip
    if mix_with_existing and ventry.clip.audio is not None:
        audio = CompositeAudioClip([ventry.clip.audio, audio])
    if audio.duration and ventry.clip.duration and \
            audio.duration > ventry.clip.duration:
        audio = audio.subclipped(0, ventry.clip.duration)
    new = ventry.clip.with_audio(audio)
    nid = _register(new, "video", ventry.label, ventry, "attach_audio",
                    parents=[ventry, aentry])
    return _describe(nid)

@mcp.tool(tags={'core'}, annotations=EDIT)
@serialized
def mix_audio_tracks(clip_ids: list[str],
                     volumes: Optional[list[float]] = None) -> dict:
    """Layer multiple audio tracks together (voice + music + SFX).

    Unlike ``concatenate`` (which plays clips one after another), this plays
    them simultaneously, all starting at t=0. Use ``extract_audio`` first if
    any input is a video's audio track. Result duration is the longest input.

    Args:
        clip_ids: Two or more audio clip_ids to mix.
        volumes: Optional per-clip volume multipliers, same length as
            clip_ids (e.g. [1.0, 0.3] to duck background music under voice).
            Defaults to 1.0 for every track.
    """
    if len(clip_ids) < 2:
        raise ValueError("Provide at least two clip_ids to mix.")
    if volumes is not None and len(volumes) != len(clip_ids):
        raise ValueError(
            f"volumes must have the same length as clip_ids "
            f"({len(clip_ids)}), got {len(volumes)}.")
    entries = [_get(cid, expect=("audio",)) for cid in clip_ids]
    tracks = []
    for i, entry in enumerate(entries):
        clip = entry.clip
        vol = volumes[i] if volumes else 1.0
        if vol != 1.0:
            clip = clip.with_volume_scaled(vol)
        tracks.append(clip)
    new = CompositeAudioClip(tracks)
    nid = _register(new, "audio", "mixed", entries[0],
                    f"mix_audio_tracks({len(clip_ids)})", parents=entries)
    return _describe(nid)


def _duck_audio(music: Any, voice: Any, duck_factor: float,
                threshold_db: float) -> Any:
    """Lower music while voice is above threshold_db. Loads both tracks into memory."""
    import numpy as np
    from moviepy import AudioArrayClip

    from moviepy_mcp.editing import _rms_db

    fps = 22050
    voice_arr = voice.to_soundarray(fps=fps)
    voice_mono = voice_arr.mean(axis=1) if voice_arr.ndim > 1 else voice_arr
    music_arr = music.to_soundarray(fps=fps)
    count = music_arr.shape[0]
    if count == 0:
        raise ValueError("Music clip has no samples to duck.")
    window = max(1, int(0.05 * fps))
    gains = np.ones(count, dtype=np.float64)
    limit = min(len(voice_mono), count)
    for start in range(0, limit, window):
        chunk = voice_mono[start:start + window]
        if _rms_db(chunk) >= threshold_db:
            gains[start:start + window] = duck_factor
    if music_arr.ndim == 1:
        ducked = (music_arr * gains).reshape(-1, 1)
    else:
        ducked = music_arr * gains[:, None]
    duration = getattr(music, "duration", None) or (count / fps)
    return AudioArrayClip(ducked, fps=fps).with_duration(duration)


@mcp.tool(tags={"core"}, annotations=EDIT)
@serialized
def duck_audio(music_clip_id: str, voice_clip_id: str,
               duck_factor: float = 0.2, threshold_db: float = -35.0) -> dict:
    """Turn music down while a voice clip is loud (sidechain ducking).

    Args:
        music_clip_id: Audio, or a video whose soundtrack should duck.
        voice_clip_id: Audio, or a video, used only as the loudness envelope.
        duck_factor: Music gain while the voice is speaking. 0.2 = 20% volume.
            1.0 leaves the music unchanged.
        threshold_db: Voice loudness (dBFS) above which the music ducks.
    """
    if not 0.0 <= duck_factor <= 1.0:
        raise ValueError(f"duck_factor must be between 0 and 1, got {duck_factor}.")
    music = _get(music_clip_id)
    voice = _get(voice_clip_id)
    music_audio = music.clip if music.kind == "audio" else music.clip.audio
    voice_audio = voice.clip if voice.kind == "audio" else voice.clip.audio
    if music_audio is None or voice_audio is None:
        raise ValueError(
            "Both clips need audio. extract_audio if you only have video."
        )
    # shortcut: the whole music and voice waveforms are loaded into RAM.
    ducked = _duck_audio(music_audio, voice_audio, duck_factor, threshold_db)
    if music.kind == "audio":
        new, kind = ducked, "audio"
    else:
        new, kind = music.clip.with_audio(ducked), music.kind
    nid = _register(
        new, kind, music.label, music,
        f"duck_audio(x{duck_factor})", parents=[music, voice],
    )
    return _describe(nid)
