"""The file outlives the engine; paths, time format and waiting for another writer."""

import sqlite3
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest

from pybodn import Collection, MemoryEngine, NewMemory
from pybodn._storage.sqlite import SQLiteStore
from pybodn.errors import StorageError, ValidationError
from tests.sqlite.support import RELEASE_AFTER, SHORT_WAIT, another_writer, run_sql


def test_records_survive_closing_and_reopening(path: Path) -> None:
    # Arrange
    with MemoryEngine.sqlite(path) as engine:
        notes = engine.collection("user:123")
        first = notes.add("Prefers Python", key="language", metadata={"b": 1, "a": [None]})
        notes.add_many([NewMemory("Likes tea", tags=["drink"]), NewMemory("Lives in Kyiv")])
        notes.update(first.id, content="Prefers Rust")  # keeps its place
        other = engine.collection("user:456").add("Prefers Go", key="language", importance=1)
        before = notes.list()

    # Act
    with MemoryEngine.sqlite(path) as engine:
        after = engine.collection("user:123").list()
        other_after = engine.collection("user:456").list()

    # Assert
    assert after == before
    assert [record.content for record in after] == ["Prefers Rust", "Likes tea", "Lives in Kyiv"]
    assert list(after[0].metadata) == ["b", "a"]
    assert all(record.created_at.tzinfo is UTC for record in after)
    assert other_after == [other]


@pytest.mark.parametrize("name", ["memory.db", "пам'ять 🙂.db"])
def test_a_path_can_be_text_or_a_path_object_with_any_letters(tmp_path: Path, name: str) -> None:
    # Arrange
    path = tmp_path / name

    # Act
    for given in (path, str(path)):
        with MemoryEngine.sqlite(given) as engine:
            engine.collection("user:123").add("Prefers Python")

    # Assert
    with MemoryEngine.sqlite(path) as engine:
        assert engine.collection("user:123").count() == 2
    assert path.exists()


def test_memory_keeps_each_engine_apart_and_nothing_after_close() -> None:
    # Act
    with MemoryEngine.sqlite(":memory:") as first, MemoryEngine.sqlite(":memory:") as second:
        record = first.collection("user:123").add("Prefers Python")

        # Assert
        assert first.collection("user:123").get(record.id) == record
        assert second.collection("user:123").count() == 0
    with MemoryEngine.sqlite(":memory:") as third:
        assert third.collection("user:123").count() == 0


def test_a_missing_folder_is_a_storage_error(tmp_path: Path) -> None:
    with pytest.raises(StorageError) as caught:
        MemoryEngine.sqlite(tmp_path / "missing" / "memory.db")
    assert isinstance(caught.value.__cause__, sqlite3.Error)


@pytest.mark.parametrize(
    ("path", "timeout", "field"),
    [
        ("", 5.0, "path"),
        (None, 5.0, "path"),
        ("memory.db", -1, "timeout"),
        ("memory.db", True, "timeout"),
    ],
)
def test_a_bad_path_or_timeout_is_rejected_before_any_file_is_made(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path: object, timeout: object, field: str
) -> None:
    # Arrange
    monkeypatch.chdir(tmp_path)

    # Act / Assert
    with pytest.raises(ValidationError, match=field):
        MemoryEngine.sqlite(path, timeout=timeout)  # type: ignore[arg-type]
    assert list(tmp_path.iterdir()) == []


def test_a_time_with_zero_microseconds_keeps_its_width_and_comes_back_unchanged(
    path: Path,
) -> None:
    # Arrange
    now = datetime(2026, 11, 13, 9, 55, tzinfo=UTC)

    # Act
    with MemoryEngine(SQLiteStore(str(path), timeout=5.0), clock=lambda: now) as engine:
        notes = engine.collection("user:123")
        record = notes.add("Prefers Python")
        stored = notes.get(record.id)

    # Assert
    assert stored == record
    assert stored is not None
    assert stored.created_at.tzinfo is UTC
    assert run_sql(path, "SELECT created_at, updated_at FROM memories") == [
        ("2026-11-13T09:55:00.000000+00:00", "2026-11-13T09:55:00.000000+00:00")
    ]


def test_a_file_uses_the_write_ahead_log_and_full_sync(path: Path) -> None:
    # Arrange
    store = SQLiteStore(str(path), timeout=5.0)

    # Act
    synchronous = store._connection.execute("PRAGMA synchronous").fetchone()[0]
    store.close()

    # Assert
    assert synchronous == 2  # FULL
    assert run_sql(path, "PRAGMA journal_mode") == [("wal",)]


def test_close_closes_the_file_so_it_can_be_removed(path: Path) -> None:
    # Arrange
    store = SQLiteStore(str(path), timeout=5.0)
    with MemoryEngine(store) as engine:
        engine.collection("user:123").add("Prefers Python")

    # Act
    path.unlink()

    # Assert
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        store._connection.execute("SELECT 1")
    assert list(path.parent.iterdir()) == []


# Each one changes the file; the record "language" exists before it runs.
WRITES: dict[str, Callable[[Collection], object]] = {
    "add": lambda notes: notes.add("Likes tea"),
    "add_many": lambda notes: notes.add_many([NewMemory("Likes tea")]),
    "put": lambda notes: notes.put("language", "Rust"),
    "update": lambda notes: notes.update(id_of_language(notes), content="Prefers Rust"),
    "delete": lambda notes: notes.delete(id_of_language(notes)),
}


def id_of_language(notes: Collection) -> str:  # a read, so it does not wait
    record = notes.get_by_key("language")
    assert record is not None
    return record.id


@pytest.mark.parametrize("write", WRITES.values(), ids=WRITES.keys())
def test_a_write_gives_up_after_the_timeout_while_another_connection_writes(
    path: Path, write: Callable[[Collection], object]
) -> None:
    with MemoryEngine.sqlite(path, timeout=SHORT_WAIT) as engine:
        # Arrange
        notes = engine.collection("user:123")
        before = notes.put("language", "Python")

        # Act
        with another_writer(path), pytest.raises(StorageError) as caught:
            write(notes)

        # Assert
        assert isinstance(caught.value.__cause__, sqlite3.OperationalError)
        assert notes.list() == [before]
        write(notes)  # works once the other connection is done


def test_a_write_waits_for_another_connection_within_the_timeout(path: Path) -> None:
    with MemoryEngine.sqlite(path) as engine, another_writer(path) as other:
        # Arrange
        notes = engine.collection("user:123")
        other.execute(
            "INSERT INTO memories (id, namespace, content, metadata_json, tags_json, importance,"
            " created_at, updated_at, version) VALUES ('0' , 'user:123', 'From elsewhere', '{}',"
            " '[]', 0.5, '2026-11-13T09:55:00.000000+00:00', '2026-11-13T09:55:00.000000+00:00', 1)"
        )
        release = threading.Timer(RELEASE_AFTER, lambda: other.execute("COMMIT"))

        # Act
        release.start()
        try:
            record = notes.add("Prefers Python")
        finally:
            release.join()

        # Assert
        assert [stored.content for stored in notes.list()] == ["From elsewhere", record.content]


def test_reads_do_not_wait_for_another_connection_that_writes(path: Path) -> None:
    with MemoryEngine.sqlite(path, timeout=SHORT_WAIT) as engine:
        # Arrange
        notes = engine.collection("user:123")
        record = notes.add("Prefers Python", key="language")

        with another_writer(path) as other:
            other.execute("DELETE FROM memories")

            # Act / Assert: the change is not committed yet, so it is not seen either
            assert notes.get(record.id) == record
            assert notes.get_by_key("language") == record
            assert notes.list() == [record]
            assert notes.count() == 1


def test_adding_nothing_does_not_wait_for_another_connection(path: Path) -> None:
    with MemoryEngine.sqlite(path, timeout=SHORT_WAIT) as engine, another_writer(path):
        assert engine.collection("user:123").add_many([]) == []


# Each one damages the record so that it no longer reads back, as another program could.
DAMAGE = {
    "metadata": "UPDATE memories SET metadata_json = '{not json'",
    # a million "[": nested deeper than Python can read back
    "too-deep": "UPDATE memories SET metadata_json = replace(hex(zeroblob(1000000)), '00', '[')",
    "tags": "UPDATE memories SET tags_json = '5'",
    "time": "UPDATE memories SET created_at = 'yesterday'",
}

# One read of a single record, and one inside a change, which must be undone.
READS: dict[str, Callable[[Collection, str], object]] = {
    "get": lambda notes, record_id: notes.get(record_id),
    "update": lambda notes, record_id: notes.update(record_id, content="Prefers Rust"),
}


@pytest.mark.parametrize("damage", DAMAGE.values(), ids=DAMAGE.keys())
@pytest.mark.parametrize("read", READS.values(), ids=READS.keys())
def test_a_damaged_record_is_a_storage_error(
    path: Path, damage: str, read: Callable[[Collection, str], object]
) -> None:
    with MemoryEngine.sqlite(path, timeout=SHORT_WAIT) as engine:
        # Arrange
        notes = engine.collection("user:123")
        record = notes.add("Prefers Python")
        run_sql(path, damage)

        # Act / Assert
        with pytest.raises(StorageError, match="damaged") as caught:
            read(notes, record.id)
        assert isinstance(caught.value.__cause__, TypeError | ValueError | RecursionError)
        notes.add("Likes tea")  # nothing was left half done
