"""Erasing deleted data: what a delete, a replacement or a purge hid is gone from the file and its
write-ahead log afterwards, and what is still stored stays. The tests read the bytes."""

import sqlite3
from collections.abc import Callable, Mapping
from contextlib import closing
from pathlib import Path

import pytest

from pybodn import Embedder, MemoryEngine, MemoryRecord
from pybodn.errors import StorageError
from tests.sqlite.support import PAST, SHORT_WAIT, format_marks, needs_read_only, run_sql

# In no other text of these tests, and no other word starts with its first letter, so the word
# index keeps it whole.
MARK = "zqvmark"
SPACE = f"{MARK}space"
# The values of the marked record that the file holds as text.
TEXTS = {
    "text": f"{MARK}text",
    "key": f"{MARK}key",
    "tag": f"{MARK}tag",
    "field": f"{MARK}field",
    "value": f"{MARK}value",
    "source": f"{MARK}source",
    "namespace": SPACE,
}
# What a new version of the record no longer holds, and everything the record left.
OLD_VALUES = frozenset({"text", "tag", "field", "value", "source"})
EVERY_TRACE = OLD_VALUES | {"id", "key", "namespace", "token"}


def add_marked(engine: MemoryEngine) -> MemoryRecord:
    """A record with the mark in every field, alone in a namespace of its own."""
    return engine.collection(SPACE).add(
        TEXTS["text"],
        key=TEXTS["key"],
        tags=[TEXTS["tag"]],
        metadata={TEXTS["field"]: TEXTS["value"]},
        source=TEXTS["source"],
    )


def file_bytes(path: Path) -> bytes:
    """The database file and its write-ahead log, as one run of bytes."""
    log = path.with_name(f"{path.name}-wal")
    return path.read_bytes() + (log.read_bytes() if log.exists() else b"")


def left_in_file(path: Path, traces: Mapping[str, bytes]) -> set[str]:
    stored = file_bytes(path)
    return {name for name, trace in traces.items() if trace in stored}


def assert_in_file(path: Path, traces: Mapping[str, bytes]) -> None:
    """Every trace is in the file, so that a test cannot pass for want of them."""
    assert left_in_file(path, traces) == set(traces)


def traces_of(record: MemoryRecord) -> dict[str, bytes]:
    """The marked record's texts, its id and its namespace's token in the word index."""
    return {name: text.encode() for name, text in TEXTS.items()} | {
        "id": record.id.encode(),
        "token": SPACE.encode().hex().encode(),
    }


type Hide = Callable[[MemoryEngine, MemoryRecord], object]


def purge(engine: MemoryEngine, record: MemoryRecord) -> None:
    engine.collection(SPACE).update(record.id, expires_at=PAST)
    engine.purge_expired()


# Each way to hide the marked record or its old values, and the traces the erase removes.
# Archived records and expired ones not yet purged are still stored, so nothing goes.
HIDE: dict[str, tuple[Hide, frozenset[str]]] = {
    "delete": (lambda engine, record: engine.collection(SPACE).delete(record.id), EVERY_TRACE),
    "purge-expired": (purge, EVERY_TRACE),
    "replacing-put": (
        lambda engine, _: engine.collection(SPACE).put(TEXTS["key"], "plain words"),
        OLD_VALUES,
    ),
    "update": (
        lambda engine, record: engine.collection(SPACE).update(
            record.id, content="plain words", tags=[], metadata={}, source=None
        ),
        OLD_VALUES,
    ),
    "archive": (lambda engine, record: engine.collection(SPACE).archive(record.id), frozenset()),
    "expiry": (
        lambda engine, record: engine.collection(SPACE).update(record.id, expires_at=PAST),
        frozenset(),
    ),
}


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


def bytes_of(texts: list[str]) -> list[list[float]]:
    """A vector of each text's first eight bytes, so that each text has a vector of its own."""
    return [[float(byte) for byte in text.encode()[:8].ljust(8, b".")] for text in texts]


def stored_vector(path: Path, record: MemoryRecord) -> bytes:
    ((vector,),) = run_sql(
        path, "SELECT vector FROM memory_vectors WHERE record_id = ?", (record.id,)
    )
    assert isinstance(vector, bytes)
    return vector


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


# How another connection holds the file: a reader keeps the log from being emptied, and a writer
# keeps the erase from starting.
HOLD = {"reader": "BEGIN", "writer": "BEGIN IMMEDIATE"}


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
