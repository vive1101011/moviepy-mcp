"""One MoviePy operation at a time, and a cancel flag exports can poll."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

LOCK = threading.RLock()

F = TypeVar("F", bound=Callable[..., Any])


def serialized(fn: F) -> F:
    """Hold the render lock for sync tools. Async tools lock inside their worker."""
    if asyncio.iscoroutinefunction(fn):
        return fn

    @wraps(fn)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with LOCK:
            return fn(*args, **kwargs)

    return wrapped  # type: ignore[return-value]
