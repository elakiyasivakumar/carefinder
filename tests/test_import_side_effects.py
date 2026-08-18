"""Importing a service must not reconfigure the process it was imported into."""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run(snippet):
    return subprocess.run(
        [sys.executable, "-c", snippet],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def test_importing_maps_service_leaves_root_logging_alone():
    """basicConfig at import time prints geocoding logs over the caller's UI."""
    result = _run(
        "import logging;"
        "before = (logging.getLogger().level, len(logging.getLogger().handlers));"
        "import services.maps_service;"
        "after = (logging.getLogger().level, len(logging.getLogger().handlers));"
        "assert before == after, f'root logging changed: {before} -> {after}'"
    )

    assert result.returncode == 0, result.stderr
