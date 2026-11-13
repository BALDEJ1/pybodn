"""What the public API cannot show about the SQLite store: a commit that fails, and
switching a file to the write-ahead log while another connection writes."""

import sqlite3
import threading
import time
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest

from pybodn import MemoryEngine
from pybodn._storage.sqlite import SQLiteStore, _use_write_ahead_log
from pybodn.errors import StorageError

SHORT_WAIT = 0.05  # seconds
RELEASE_AFTER = 0.2  # seconds after which the other connection stops writing
GIVE_UP_WITHIN = 5  # seconds: generous for a slow machine, far below a wait that never ends


class FailingCommit:
    """Stands in for the store's connection. COMMIT fails, as it can on a full disk."""

    def __init__(self, connection: sqlite3.Connection, *, rolls_back: bool) -> None:
        self._connection = connection
        self._rolls_back = rolls_back  # SQLite rolls back by itself after some failures

    @property
    def in_transaction(self) -> bool:
        return self._connection.in_transaction

    def execute(self, sql: str, *args: Any) -> sqlite3.Cursor:
        if sql == "COMMIT":
            if self._rolls_back:
                self._connection.execute("ROLLBACK")
            raise sqlite3.OperationalError("disk I/O error")
        return self._connection.execute(sql, *args)

    def executemany(self, sql: str, *args: Any) -> sqlite3.Cursor:
        return self._connection.executemany(sql, *args)


@pytest.mark.parametrize("rolls_back", [False, True], ids=["still-open", "already-rolled-back"])
def test_a_failed_commit_writes_nothing_and_leaves_the_store_usable(
    tmp_path: Path, rolls_back: bool
) -> None:
    # Arrange
    store = SQLiteStore(str(tmp_path / "memory.db"), timeout=5.0)
    connection = store._connection
    with MemoryEngine(store) as engine:
        notes = engine.collection("user:123")
        store._connection = FailingCommit(connection, rolls_back=rolls_back)  # type: ignore[assignment]

        # Act
        with pytest.raises(StorageError) as caught:
            notes.add("Prefers Python")

        # Assert
        assert isinstance(caught.value.__cause__, sqlite3.OperationalError)
        assert not connection.in_transaction
        store._connection = connection
        assert notes.count() == 0
        notes.add("Prefers Python")
        assert notes.count() == 1


def test_the_switch_to_the_log_waits_for_another_writer(tmp_path: Path) -> None:
    # Arrange: SQLite itself refuses the switch at once while another connection writes
    path = tmp_path / "memory.db"
    with (
        closing(sqlite3.connect(path, autocommit=True)) as connection,
        closing(sqlite3.connect(path, autocommit=True, check_same_thread=False)) as other,
    ):
        connection.execute("CREATE TABLE notes (text)")
        other.execute("BEGIN IMMEDIATE")
        release = threading.Timer(RELEASE_AFTER, lambda: other.execute("ROLLBACK"))

        # Act
        release.start()
        try:
            _use_write_ahead_log(connection, timeout=5.0)
        finally:
            release.join()

        # Assert
        assert connection.execute("PRAGMA journal_mode").fetchall() == [("wal",)]


def test_the_switch_to_the_log_gives_up_after_the_timeout(tmp_path: Path) -> None:
    # Arrange
    path = tmp_path / "memory.db"
    with (
        closing(sqlite3.connect(path, autocommit=True)) as connection,
        closing(sqlite3.connect(path, autocommit=True)) as other,
    ):
        connection.execute("CREATE TABLE notes (text)")
        other.execute("BEGIN IMMEDIATE")

        # Act
        started = time.monotonic()
        try:
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                _use_write_ahead_log(connection, timeout=SHORT_WAIT)
        finally:
            waited = time.monotonic() - started
            other.execute("ROLLBACK")

        # Assert
        assert SHORT_WAIT <= waited < GIVE_UP_WITHIN
        assert connection.execute("PRAGMA journal_mode").fetchall() == [("delete",)]
