"""What the public API cannot show about the SQLite store: a commit that fails."""

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from pybodn import MemoryEngine
from pybodn._storage.sqlite import SQLiteStore
from pybodn.errors import StorageError


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
