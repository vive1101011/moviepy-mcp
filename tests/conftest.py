import os
import tempfile

# Force a throwaway workspace before the server reads settings.
os.environ["MOVIEPY_MCP_MEDIA_ROOT"] = tempfile.mkdtemp(prefix="moviepy-mcp-tests-")
os.environ["MOVIEPY_MCP_PERSIST"] = "0"
os.environ["MOVIEPY_MCP_ADVANCED"] = "1"
os.environ["MOVIEPY_MCP_ISOLATE_SESSIONS"] = "0"

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
