"""The fast erase: every engine zeroes what a delete, a replacement or a purge leaves, so
``erase_deleted(fast=True)`` only rebuilds the word index and empties the log. It is full the first
time on a file, and whenever a full erase has not finished since. The tests read the bytes."""

import re
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from pybodn import Embedder, MemoryEngine, NewMemory
from pybodn._storage import sqlite as sqlite_store
from pybodn.errors import StorageError
from tests.sqlite.support import (
    HIDE,
    HOLD,
    MARK,
    PAST,
    SHORT_WAIT,
    SPACE,
    Hide,
    add_marked,
    assert_in_file,
    bytes_of,
    left_in_file,
    run_sql,
    stored_vector,
    traces_of,
)

# Enough text to fill whole pages, so that deleting it frees some.
FILLER = " ".join(["filler"] * 150)
BULK = 300
# What a delete leaves in the word index until the index is rebuilt: the record's words and its
# namespace's token. Everything else of the record is zeroed by the delete itself.
IN_WORD_INDEX = frozenset({"text", "token"})


def free_pages(path: Path) -> int:
    """How many pages of the file are free. A full erase leaves none."""
    ((count,),) = run_sql(path, "PRAGMA freelist_count")
    assert isinstance(count, int)
    return count


def free_some_pages(engine: MemoryEngine) -> None:
    """Add records that fill whole pages, then delete them, so the file has pages to free."""
    notes = engine.collection("bulk")
    for record in notes.add_many([NewMemory(f"{FILLER} {n}") for n in range(BULK)]):
        notes.delete(record.id)


def move_log(path: Path) -> None:
    """Move the log into the file and empty it, as a checkpoint of another connection does."""
    ((busy, _, _),) = run_sql(path, "PRAGMA wal_checkpoint(TRUNCATE)")
    assert busy == 0


def stop_zeroing(engine: MemoryEngine) -> None:
    """Make ``engine`` delete as pybodn 1.0 does: without zeroing what it deletes."""
    connection = engine._store._connection  # type: ignore[attr-defined]
    connection.execute("PRAGMA secure_delete = OFF").fetchall()


def test_a_delete_zeroes_the_record_in_the_file_at_once(path: Path) -> None:
    with MemoryEngine.sqlite(path) as engine:
        # Arrange
        record = add_marked(engine)
        traces = traces_of(record)
        move_log(path)
        assert_in_file(path, traces)

        # Act
        engine.collection(SPACE).delete(record.id)
        move_log(path)

        # Assert: only the word index still holds it, until an erase rebuilds the index
        left = left_in_file(path, traces)
    assert left <= IN_WORD_INDEX


@pytest.mark.parametrize(("hide", "erased"), HIDE.values(), ids=HIDE.keys())
def test_a_fast_erase_removes_from_the_file_only_what_was_hidden(
    path: Path, hide: Hide, erased: frozenset[str]
) -> None:
    with MemoryEngine.sqlite(path) as engine:
        # Arrange: a full erase first, so the next one is fast
        engine.erase_deleted()
        engine.collection("user:123").add("prefers python for backend work")
        record = add_marked(engine)
        traces = traces_of(record)
        assert_in_file(path, traces)
        hide(engine, record)
        free_some_pages(engine)

        # Act
        engine.erase_deleted(fast=True)

        # Assert, before the close, which cleans up the log by itself
        left = left_in_file(path, traces)
        free = free_pages(path)
    assert left == set(traces) - erased
    assert free > 0  # it did not rewrite the file
    assert run_sql(path, "PRAGMA journal_mode") == [("wal",)]
    assert run_sql(path, "PRAGMA integrity_check") == [("ok",)]


def test_the_first_fast_erase_of_a_file_is_full_and_the_next_is_fast(path: Path) -> None:
    with MemoryEngine.sqlite(path) as engine:
        # Act
        free_some_pages(engine)
        engine.erase_deleted(fast=True)
        after_first = free_pages(path)
        free_some_pages(engine)
        engine.erase_deleted(fast=True)
        after_second = free_pages(path)

    # Assert
    assert after_first == 0
    assert after_second > 0


def test_a_full_erase_whose_rewrite_fails_leaves_the_next_fast_one_full(
    path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with MemoryEngine.sqlite(path) as engine:
        # Arrange: a rewrite that fails, as on a full disk
        free_some_pages(engine)
        with monkeypatch.context() as patch:
            patch.setattr(sqlite_store, "_VACUUM", "VACUUM missing")
            with pytest.raises(StorageError, match="deleted data may still be in the file"):
                engine.erase_deleted()

        # Act
        engine.erase_deleted(fast=True)

        # Assert
        free = free_pages(path)
    assert free == 0


def test_a_full_erase_that_cannot_empty_the_log_still_lets_a_fast_one_erase_what_came_before(
    path: Path,
) -> None:
    with MemoryEngine.sqlite(path, timeout=SHORT_WAIT) as engine:
        # Arrange: a record deleted without zeroing, as pybodn 1.0 deletes, then a full erase
        # that rewrites the file but that a reader keeps from emptying the log
        record = add_marked(engine)
        traces = traces_of(record)
        with MemoryEngine.sqlite(path) as old:
            stop_zeroing(old)
            old.collection(SPACE).delete(record.id)
        assert left_in_file(path, traces) - IN_WORD_INDEX
        with closing(sqlite3.connect(path, autocommit=True)) as reader:
            reader.execute("BEGIN")
            reader.execute("SELECT count(*) FROM memories").fetchall()
            with pytest.raises(StorageError, match="deleted data may still be in the file"):
                engine.erase_deleted()
        free_some_pages(engine)

        # Act
        engine.erase_deleted(fast=True)

        # Assert: it was fast, and emptying the log carried the rewrite into the file
        left = left_in_file(path, traces)
        free = free_pages(path)
    assert left == set()
    assert free > 0


@pytest.mark.parametrize("begin", HOLD.values(), ids=HOLD.keys())
def test_a_fast_erase_fails_while_another_connection_holds_the_file_and_succeeds_when_repeated(
    path: Path, begin: str
) -> None:
    with MemoryEngine.sqlite(path, timeout=SHORT_WAIT) as engine:
        # Arrange
        engine.erase_deleted()
        record = add_marked(engine)
        traces = traces_of(record)
        assert_in_file(path, traces)
        engine.collection(SPACE).delete(record.id)

        # Act
        with closing(sqlite3.connect(path, autocommit=True)) as other:
            other.execute(begin)
            other.execute("SELECT count(*) FROM memories").fetchall()
            with pytest.raises(StorageError, match="deleted data may still be in the file"):
                engine.erase_deleted(fast=True)
        engine.erase_deleted(fast=True)

        # Assert
        left = left_in_file(path, traces)
    assert left == set()


def test_a_fast_erase_misses_what_a_writer_without_zeroing_deleted_and_a_full_one_does_not(
    path: Path,
) -> None:
    with MemoryEngine.sqlite(path) as engine:
        # Arrange: a writer that deletes as pybodn 1.0 does, after the full erase
        engine.erase_deleted()
        record = add_marked(engine)
        traces = traces_of(record)
        with MemoryEngine.sqlite(path) as old:
            stop_zeroing(old)
            old.collection(SPACE).delete(record.id)
        assert left_in_file(path, traces) - IN_WORD_INDEX

        # Act
        engine.erase_deleted(fast=True)
        after_fast = left_in_file(path, traces)
        engine.erase_deleted()
        after_full = left_in_file(path, traces)

    # Assert: the README says to erase in full when an older release writes to the file
    assert after_fast - IN_WORD_INDEX
    assert after_full == set()


def test_an_sqlite_that_cannot_zero_is_refused_and_the_file_stays_as_it_was(
    path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: a marked file, then an SQLite that ignores the request to zero
    with MemoryEngine.sqlite(path) as engine:
        engine.erase_deleted()
    before = path.read_bytes()
    monkeypatch.setattr(sqlite_store, "_ZERO_DELETED", "PRAGMA secure_delete = OFF")

    # Act / Assert
    with pytest.raises(StorageError):
        MemoryEngine.sqlite(path)
    assert path.read_bytes() == before


def test_a_fast_erase_removes_the_vector_of_a_deleted_record_and_one_replaced(path: Path) -> None:
    pytest.importorskip("numpy")  # vectors need NumPy
    shorter = Embedder("bytes-short", lambda texts: [vector[:4] for vector in bytes_of(texts)])
    with MemoryEngine.sqlite(path, embedder=Embedder("bytes", bytes_of)) as engine:
        # Arrange: one record deleted, the other given a vector of another model and size
        engine.erase_deleted()
        notes = engine.collection(SPACE)
        deleted, kept = notes.add(f"one {MARK}"), notes.add(f"two {MARK}")
        vectors = {record.id: stored_vector(path, record) for record in (deleted, kept)}
        assert_in_file(path, vectors)
        notes.delete(deleted.id)
        with MemoryEngine.sqlite(path, embedder=shorter) as other:
            other.embed_missing()

        # Act
        engine.erase_deleted(fast=True)

        # Assert
        left = left_in_file(path, vectors)
    assert left == set()


# A file with deep indexes, long texts that spill over pages, and every way to hide a record,
# so that a trace kept where pages split, merge or spill would show.
DEEP = 3000
LONG_KEY = "x" * 480  # long keys make the key index deep with few records
LONG_TEXT = " ".join(["spill"] * 1800)  # longer than two pages
TRACE = re.compile(rb"zq[kw][dlon]\d{5}")


def deep_key(state: str, n: int) -> str:
    return f"zqk{state}{n:05d}{LONG_KEY}"


def deep_word(state: str, n: int) -> str:
    return f"zqw{state}{n:05d}"


def fate(n: int) -> str:
    """What happens to record ``n``: deleted, purged, replaced, or kept."""
    if 1000 <= n < 1500 or n % 2:
        return "delete"
    return {4: "purge", 2: "replace"}.get(n % 10, "keep")


def deep_traces(path: Path) -> set[bytes]:
    files = [path, path.with_name(f"{path.name}-wal"), path.with_name(f"{path.name}-shm")]
    stored = b"".join(file.read_bytes() for file in files if file.exists())
    return set(TRACE.findall(stored))


def test_a_fast_erase_leaves_no_trace_in_a_file_with_deep_indexes(path: Path) -> None:
    dead: set[bytes] = set()
    live: set[bytes] = set()
    with MemoryEngine.sqlite(path) as engine:
        # Arrange
        engine.erase_deleted()
        notes = engine.collection("deep")
        memories = []
        for n in range(DEEP):
            state = "o" if fate(n) == "replace" else ("l" if fate(n) == "keep" else "d")
            text = deep_word(state, n) + (f" {LONG_TEXT}" if n % 25 == 0 else "")
            expiry = PAST if fate(n) == "purge" else None
            key_state = "l" if fate(n) in ("keep", "replace") else "d"
            memories.append(NewMemory(text, key=deep_key(key_state, n), expires_at=expiry))
            (live if key_state == "l" else dead).add(deep_key(key_state, n)[:9].encode())
            (live if state == "l" else dead).add(deep_word(state, n).encode())
        records = notes.add_many(memories)
        move_log(path)
        assert dead | live <= deep_traces(path)
        for n, record in enumerate(records):
            if fate(n) == "delete":
                notes.delete(record.id)
            elif fate(n) == "replace":
                notes.update(record.id, content=deep_word("n", n))
                live.add(deep_word("n", n).encode())
        engine.purge_expired()

        # Act
        engine.erase_deleted(fast=True)

        # Assert
        found = deep_traces(path)
        free = free_pages(path)
    assert found & dead == set()
    assert live <= found
    assert free > 0
    assert run_sql(path, "PRAGMA integrity_check") == [("ok",)]
