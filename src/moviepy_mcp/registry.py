"""Session clip registry, size caps, and a project file of source clips."""

from __future__ import annotations

import atexit
import json
import logging
import os
import re
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Optional

from moviepy_mcp.config import get_settings
from moviepy_mcp.models import ClipInfo

logger = logging.getLogger("moviepy_mcp.registry")

# shortcut: replay reopens load/upload/download files. Effect clips are recorded
# in the project file and reported by project_status; they are not rebuilt.


@dataclass
class ClipEntry:
    clip: Any
    kind: str
    label: str
    history: list[str] = field(default_factory=list)
    parent_ids: list[str] = field(default_factory=list)
    op: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    source_path: Optional[str] = None


_SESSIONS: dict[str, dict[str, ClipEntry]] = {}
_RESTORED: set[str] = set()
_SKIPPED: dict[str, list[str]] = {}
_SOURCE_OPS = {"load_video", "load_audio", "load_image", "upload_media", "download_video"}


def session_key() -> str:
    settings = get_settings()
    if not settings.isolate_sessions:
        return "default"
    try:
        from fastmcp.server.dependencies import get_context

        return str(get_context().session_id)
    except Exception:
        return "default"


def _safe_name(sid: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", sid)[:80] or "session"


def _project_path(sid: str) -> str:
    folder = os.path.join(str(get_settings().media_root), ".moviepy-mcp")
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"{_safe_name(sid)}.json")


def _id_of(bucket: dict[str, ClipEntry], entry: ClipEntry) -> Optional[str]:
    for clip_id, candidate in bucket.items():
        if candidate is entry:
            return clip_id
    return None


def _close(entry: ClipEntry) -> None:
    try:
        entry.clip.close()
    except Exception:
        pass


def _check_limits(clip: Any, kind: str) -> None:
    settings = get_settings()
    if kind in ("video", "image"):
        width = int(getattr(clip, "w", 0) or 0)
        height = int(getattr(clip, "h", 0) or 0)
        if width > settings.max_width or height > settings.max_height:
            raise ValueError(
                f"Clip is {width}x{height}, over the limit "
                f"{settings.max_width}x{settings.max_height}. "
                "Resize first, or raise MOVIEPY_MCP_MAX_WIDTH / MAX_HEIGHT."
            )
    duration = getattr(clip, "duration", None) or 0
    if duration > settings.max_duration_seconds:
        raise ValueError(
            f"Clip duration {duration:.1f}s is over the limit "
            f"{settings.max_duration_seconds:.0f}s. Trim it, or raise "
            "MOVIEPY_MCP_MAX_DURATION."
        )


def _evict(bucket: dict[str, ClipEntry]) -> None:
    """Drop the oldest clip nothing else references, until under the cap."""
    settings = get_settings()
    while len(bucket) >= settings.max_clips:
        referenced = {pid for entry in bucket.values() for pid in entry.parent_ids}
        victim = next((cid for cid in bucket if cid not in referenced), None)
        if victim is None:
            raise ValueError(
                f"Registry is full ({settings.max_clips} clips) and every clip "
                "is still an input to another. delete_clip some ids first."
            )
        _close(bucket.pop(victim))


def _persist(sid: str) -> None:
    if not get_settings().persist:
        return
    records = []
    for clip_id, entry in _SESSIONS.get(sid, {}).items():
        records.append({
            "id": clip_id,
            "kind": entry.kind,
            "label": entry.label,
            "history": entry.history,
            "parent_ids": entry.parent_ids,
            "op": entry.op,
            "args": entry.args,
            "source_path": entry.source_path,
        })
    path = _project_path(sid)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"version": 1, "clips": records}, fh)
    os.replace(tmp, path)


def _reopen(record: dict[str, Any]) -> Any:
    from moviepy import AudioFileClip, ImageClip, VideoFileClip

    path = record.get("source_path")
    if not path or not os.path.isfile(path):
        return None
    kind = record.get("kind")
    op = record.get("op")
    if kind == "audio" or op == "load_audio":
        return AudioFileClip(path)
    if kind == "image" or op == "load_image":
        duration = (record.get("args") or {}).get("duration_seconds") or 5.0
        return ImageClip(path).with_duration(duration)
    return VideoFileClip(path)


def _restore(sid: str) -> None:
    if sid in _RESTORED or not get_settings().persist:
        _RESTORED.add(sid)
        return
    _RESTORED.add(sid)
    path = _project_path(sid)
    if not os.path.isfile(path):
        return
    try:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read project file %s: %s", path, exc)
        return
    bucket = _SESSIONS.setdefault(sid, {})
    skipped: list[str] = []
    for record in payload.get("clips") or []:
        clip_id = record.get("id")
        if not clip_id or clip_id in bucket:
            continue
        if record.get("op") not in _SOURCE_OPS:
            skipped.append(str(clip_id))
            continue
        try:
            clip = _reopen(record)
        except Exception as exc:
            logger.warning("Could not restore %s: %s", clip_id, exc)
            skipped.append(str(clip_id))
            continue
        if clip is None:
            skipped.append(str(clip_id))
            continue
        bucket[clip_id] = ClipEntry(
            clip=clip,
            kind=record.get("kind") or "video",
            label=record.get("label") or clip_id,
            history=list(record.get("history") or []),
            parent_ids=list(record.get("parent_ids") or []),
            op=record.get("op") or "",
            args=dict(record.get("args") or {}),
            source_path=record.get("source_path"),
        )
    _SKIPPED[sid] = skipped
    if skipped:
        logger.info("Restored sources for %s; %s derived clips were not rebuilt", sid, len(skipped))


def _trim_sessions(current: str) -> None:
    settings = get_settings()
    while len(_SESSIONS) > settings.max_sessions:
        victim = next((sid for sid in _SESSIONS if sid != current), None)
        if victim is None:
            return
        for entry in _SESSIONS.pop(victim).values():
            _close(entry)
        _RESTORED.discard(victim)
        _SKIPPED.pop(victim, None)


def bucket_for(sid: str) -> dict[str, ClipEntry]:
    if sid not in _SESSIONS:
        _SESSIONS[sid] = {}
        _trim_sessions(sid)
        _restore(sid)
    return _SESSIONS[sid]


def _bucket() -> dict[str, ClipEntry]:
    return bucket_for(session_key())


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _get(clip_id: str, expect: Optional[tuple[str, ...]] = None) -> ClipEntry:
    bucket = _bucket()
    entry = bucket.get(clip_id)
    if entry is None and session_key() != "default":
        entry = _SESSIONS.get("default", {}).get(clip_id)
    if entry is None:
        known = ", ".join(bucket) or "(registry is empty)"
        raise ValueError(
            f"Unknown clip_id '{clip_id}'. Known clip_ids: {known}. "
            "Load media first with load_video/load_audio/load_image."
        )
    if expect and entry.kind not in expect:
        raise ValueError(
            f"clip_id '{clip_id}' is a {entry.kind} clip but this tool "
            f"requires one of: {', '.join(expect)}."
        )
    return entry


def _register(
    clip: Any,
    kind: str,
    label: str,
    parent: Optional[ClipEntry],
    op: str,
    *,
    parents: Optional[list[ClipEntry]] = None,
    args: Optional[dict[str, Any]] = None,
    source_path: Optional[str] = None,
    clip_id: Optional[str] = None,
) -> str:
    try:
        _check_limits(clip, kind)
    except ValueError:
        _close(ClipEntry(clip=clip, kind=kind, label=label))
        raise
    sid = session_key()
    bucket = bucket_for(sid)
    _evict(bucket)
    parent_entries = parents if parents is not None else ([parent] if parent else [])
    parent_ids = [pid for entry in parent_entries if (pid := _id_of(bucket, entry))]
    history_parents = [bucket[pid].history for pid in parent_ids if pid in bucket]
    history = (history_parents[0][:] if history_parents else []) + [op]
    clip_id = clip_id or _new_id(kind)
    bucket[clip_id] = ClipEntry(
        clip=clip,
        kind=kind,
        label=label,
        history=history,
        parent_ids=parent_ids,
        op=op.split("(", 1)[0],
        args=args or {},
        source_path=source_path,
    )
    _persist(sid)
    return clip_id


def _describe(clip_id: str) -> dict[str, Any]:
    entry = _get(clip_id)
    clip = entry.clip
    info: dict[str, Any] = {
        "clip_id": clip_id,
        "kind": entry.kind,
        "label": entry.label,
        "duration_seconds": round(clip.duration, 3) if clip.duration else None,
        "history": entry.history,
    }
    if entry.kind in ("video", "image"):
        info["size"] = {"width": int(clip.w), "height": int(clip.h)}
        fps = getattr(clip, "fps", None)
        info["fps"] = float(fps) if fps is not None else None
        info["has_audio"] = clip.audio is not None
    ClipInfo.model_validate(info)
    return info


def current_ids() -> list[str]:
    return list(_bucket())


def remove(clip_id: str) -> int:
    """Drop one clip from whichever session holds it and return the current count."""
    _get(clip_id)
    for sid, bucket in _SESSIONS.items():
        entry = bucket.pop(clip_id, None)
        if entry is None:
            continue
        _close(entry)
        _persist(sid)
        return len(_bucket())
    raise ValueError(f"Unknown clip_id '{clip_id}'.")


def project_report() -> dict[str, Any]:
    sid = session_key()
    bucket = bucket_for(sid)
    return {
        "session": sid,
        "clips": len(bucket),
        "restored_source_ids": [
            cid for cid, entry in bucket.items() if entry.op in _SOURCE_OPS
        ],
        "not_rebuilt": list(_SKIPPED.get(sid, [])),
    }


def reset_all() -> None:
    """Close every clip and drop session state. Used by tests and process exit."""
    for bucket in list(_SESSIONS.values()):
        for entry in list(bucket.values()):
            _close(entry)
        bucket.clear()
    _SESSIONS.clear()
    _RESTORED.clear()
    _SKIPPED.clear()


class _RegistryView:
    """Dict-like view of every open clip. Tests clear this between cases."""

    def clear(self) -> None:
        reset_all()

    def __contains__(self, clip_id: object) -> bool:
        return any(clip_id in bucket for bucket in _SESSIONS.values())

    def __iter__(self) -> Iterator[str]:
        for bucket in _SESSIONS.values():
            yield from bucket

    def values(self) -> list[ClipEntry]:
        return [entry for bucket in _SESSIONS.values() for entry in bucket.values()]

    def __len__(self) -> int:
        return sum(len(bucket) for bucket in _SESSIONS.values())


_REGISTRY = _RegistryView()


atexit.register(reset_all)
