"""The metadata payload every editing tool returns."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ClipSize(BaseModel):
    width: int
    height: int


class ClipInfo(BaseModel):
    clip_id: str
    kind: str
    label: str
    duration_seconds: Optional[float] = None
    history: list[str] = Field(default_factory=list)
    size: Optional[ClipSize] = None
    fps: Optional[float] = None
    has_audio: Optional[bool] = None
