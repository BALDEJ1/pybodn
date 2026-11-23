"""Erasing deleted data: what a delete, a replacement or a purge hid is gone from the file and its
write-ahead log afterwards, and what is still stored stays. The tests read the bytes."""

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from pybodn import Embedder, MemoryEngine
from pybodn.errors import StorageError
from tests.sqlite.support import (
    HIDE,
    HOLD,
    MARK,
    SHORT_WAIT,
    SPACE,
    Hide,
    add_marked,
    assert_in_file,
    bytes_of,
    format_marks,
    left_in_file,
    needs_read_only,
    run_sql,
    stored_vector,
    traces_of,
)


@pytest.mark.parametrize(("hide", "erased"), HIDE.values(), ids=HIDE.keys())
def test_erase_deleted_removes_from_the_file_only_what_was_hidden(
    path: Path, hide: Hide, erased: frozenset[str]
) -> None:
    with MemoryEngine.sqlite(path) as engine:
        # Arrange: another record that stays, and the marked one, hidden
        engine.collection("user:123").add("prefers python for backend work")
        record = add_marked(engine)
        traces = traces_of(record)
        assert_in_file(path, traces)
        marks = format_marks(path)
        hide(engine, record)

        # Act
        engine.erase_deleted()

        # Assert, before the close, which cleans up the log by itself
        left = left_in_file(path, traces)
    assert left == set(traces) - erased
    assert format_marks(path) == marks
    assert run_sql(path, "PRAGMA journal_mode") == [("wal",)]
    assert run_sql(path, "PRAGMA integrity_check") == [("ok",)]


def test_erase_deleted_removes_the_vector_of_a_deleted_record_and_one_replaced(
    path: Path,
) -> None:
    pytest.importorskip("numpy")  # vectors need NumPy
    shorter = Embedder("bytes-short", lambda texts: [vector[:4] for vector in bytes_of(texts)])
    with MemoryEngine.sqlite(path, embedder=Embedder("bytes", bytes_of)) as engine:
        # Arrange: one record deleted, the other given a vector of another model and size, as
        # moving to a new model does
        notes = engine.collection(SPACE)
        deleted, kept = notes.add(f"one {MARK}"), notes.add(f"two {MARK}")
        vectors = {record.id: stored_vector(path, record) for record in (deleted, kept)}
        assert_in_file(path, vectors)
        notes.delete(deleted.id)
        with MemoryEngine.sqlite(path, embedder=shorter) as other:
            other.embed_missing()

        # Act
        engine.erase_deleted()

        # Assert
        left = left_in_file(path, vectors)
    assert left == set()


@pytest.mark.parametrize("begin", HOLD.values(), ids=HOLD.keys())
def test_erase_deleted_fails_while_another_connection_holds_the_file_and_succeeds_when_repeated(
    path: Path, begin: str
) -> None:
    with MemoryEngine.sqlite(path, timeout=SHORT_WAIT) as engine:
        # Arrange
        record = add_marked(engine)
        traces = traces_of(record)
        assert_in_file(path, traces)
        marks = format_marks(path)
        engine.collection(SPACE).delete(record.id)

        # Act
        with closing(sqlite3.connect(path, autocommit=True)) as other:
            other.execute(begin)
            other.execute("SELECT count(*) FROM memories").fetchall()
            with pytest.raises(StorageError, match="deleted data may still be in the file"):
                engine.erase_deleted()
            assert format_marks(path) == marks
            assert run_sql(path, "PRAGMA journal_mode") == [("wal",)]
        engine.erase_deleted()

        # Assert
        left = left_in_file(path, traces)
    assert left == set()


@needs_read_only
@pytest.mark.parametrize("fast", [False, True], ids=["full", "fast"])
def test_erase_deleted_on_a_read_only_file_fails_and_leaves_it_as_it_was(
    path: Path, fast: bool
) -> None:
    # Arrange
    with MemoryEngine.sqlite(path) as engine:
        engine.collection(SPACE).delete(add_marked(engine).id)
    before = path.read_bytes()
    path.chmod(0o444)

    # Act
    try:
        with MemoryEngine.sqlite(path, timeout=SHORT_WAIT) as engine:
            with pytest.raises(StorageError):
                engine.erase_deleted(fast=fast)
    finally:
        path.chmod(0o644)

    # Assert
    assert path.read_bytes() == before
