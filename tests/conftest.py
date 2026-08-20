import pytest

from moviepy_mcp.server import _REGISTRY


@pytest.fixture(autouse=True)
def clear_registry():
    _REGISTRY.clear()
    yield
    for entry in list(_REGISTRY.values()):
        try:
            entry.clip.close()
        except Exception:
            pass
    _REGISTRY.clear()
