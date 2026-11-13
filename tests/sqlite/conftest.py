"""What only the SQLite store has: a file that outlives the engine and other connections to it."""

from pathlib import Path

import pytest


@pytest.fixture
def path(tmp_path: Path) -> Path:
    """A database file in a folder of its own. It does not exist yet."""
    return tmp_path / "memory.db"
