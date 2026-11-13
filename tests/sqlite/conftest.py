"""What only the SQLite store has: a file that outlives the engine and other connections to it."""

import sqlite3
from collections.abc import Callable
from contextlib import closing
from pathlib import Path

import pytest


@pytest.fixture
def path(tmp_path: Path) -> Path:
    """A database file in a folder of its own. It does not exist yet."""
    return tmp_path / "memory.db"


@pytest.fixture
def run_sql(path: Path) -> Callable[[str], list[tuple[object, ...]]]:
    """Run one statement on the file through a connection of its own, as another program would."""

    def run(sql: str) -> list[tuple[object, ...]]:
        with closing(sqlite3.connect(path, autocommit=True)) as connection:
            return connection.execute(sql).fetchall()

    return run
