"""Two engines on one file act as two programs would: each sees the other's writes at once.
The last test runs two real processes."""

import multiprocessing
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from pybodn import MemoryEngine, NewMemory
from pybodn.errors import ConcurrentUpdateError, RecordNotFoundError
from tests.sqlite.support import (
    SHORT_WAIT,
    THREADS,
    WAIT_SECONDS,
    add_in_another_process,
    run_sql,
)

CALLS = 20
MISSING_ID = "0" * 32


def test_a_write_in_one_engine_is_seen_at_once_in_the_other(path: Path) -> None:
    with MemoryEngine.sqlite(path) as first, MemoryEngine.sqlite(path) as second:
        mine, theirs = first.collection("user:123"), second.collection("user:123")
        assert theirs.count() == 0  # a read that left anything open would hide what follows

        record = mine.add("Prefers Python", key="language")
        assert theirs.get(record.id) == record
        assert theirs.get_by_key("language") == record

        changed = theirs.update(record.id, content="Prefers Rust")
        assert mine.get(record.id) == changed
        assert mine.list() == [changed]

        assert theirs.delete(record.id) is True
        assert mine.get(record.id) is None
        assert mine.count() == 0


def test_both_engines_share_one_order_of_records(path: Path) -> None:
    # Arrange
    with MemoryEngine.sqlite(path) as first, MemoryEngine.sqlite(path) as second:
        mine, theirs = first.collection("user:123"), second.collection("user:123")

        # Act
        added = [mine.add("first"), theirs.add("second"), mine.add("third")]
        theirs.put("key", "fourth")

        # Assert
        expected = [*added, theirs.get_by_key("key")]
        assert mine.list() == expected
        assert theirs.list() == expected


def test_an_update_with_a_version_changed_by_the_other_engine_is_refused(path: Path) -> None:
    with MemoryEngine.sqlite(path) as first, MemoryEngine.sqlite(path) as second:
        # Arrange
        mine, theirs = first.collection("user:123"), second.collection("user:123")
        record = mine.add("Prefers Python")
        theirs.update(record.id, content="Prefers Rust", expected_version=record.version)

        # Act / Assert
        with pytest.raises(ConcurrentUpdateError):
            mine.update(record.id, content="Prefers Go", expected_version=record.version)
        stored = theirs.get(record.id)
        assert stored is not None
        assert (stored.content, stored.version) == ("Prefers Rust", 2)


def test_a_failed_change_in_one_engine_leaves_the_file_free_for_both(path: Path) -> None:
    with (
        MemoryEngine.sqlite(path, timeout=SHORT_WAIT) as first,
        MemoryEngine.sqlite(path, timeout=SHORT_WAIT) as second,
    ):
        # Arrange
        mine, theirs = first.collection("user:123"), second.collection("user:123")
        record = mine.add("Prefers Python")

        # Act
        with pytest.raises(RecordNotFoundError):
            mine.update(MISSING_ID, content="Prefers Rust")
        with pytest.raises(ConcurrentUpdateError):
            mine.update(record.id, content="Prefers Rust", expected_version=2)

        # Assert: neither waits, as it would for a change left open
        theirs.add("Likes tea")
        mine.add("Lives in Kyiv")
        assert theirs.count() == 3


def test_parallel_puts_from_two_engines_lose_nothing(path: Path) -> None:
    # Arrange
    start = threading.Barrier(THREADS, timeout=WAIT_SECONDS)
    with (
        MemoryEngine.sqlite(path, timeout=WAIT_SECONDS) as first,
        MemoryEngine.sqlite(path, timeout=WAIT_SECONDS) as second,
    ):

        def put_many(engine: MemoryEngine) -> None:
            notes = engine.collection("user:123")
            start.wait()
            for _ in range(CALLS):
                notes.put("counter", "value")

        # Act
        with ThreadPoolExecutor(THREADS) as pool:
            futures = [pool.submit(put_many, (first, second)[n % 2]) for n in range(THREADS)]
        for future in futures:
            future.result()

        # Assert
        record = second.collection("user:123").get_by_key("counter")
        assert record is not None
        assert record.version == THREADS * CALLS
        assert first.collection("user:123").count() == 1


def test_a_text_one_engine_added_is_a_copy_for_the_other(path: Path) -> None:
    with MemoryEngine.sqlite(path) as first, MemoryEngine.sqlite(path) as second:
        # Arrange
        mine, theirs = first.collection("user:123"), second.collection("user:123")
        record = mine.add("Prefers Python")

        # Act
        again = theirs.add("Prefers Python", dedupe=True)
        in_a_batch, new = theirs.add_many(
            [NewMemory("Prefers Python"), NewMemory("Likes tea")], dedupe=True
        )

        # Assert
        assert again == in_a_batch == record
        assert mine.list() == [record, new]


def test_parallel_adds_of_one_text_from_two_engines_write_it_once(path: Path) -> None:
    # Arrange
    start = threading.Barrier(THREADS, timeout=WAIT_SECONDS)
    with (
        MemoryEngine.sqlite(path, timeout=WAIT_SECONDS) as first,
        MemoryEngine.sqlite(path, timeout=WAIT_SECONDS) as second,
    ):

        def add(engine: MemoryEngine) -> str:
            notes = engine.collection("user:123")
            start.wait()
            return notes.add("Prefers Python", dedupe=True).id

        # Act
        with ThreadPoolExecutor(THREADS) as pool:
            futures = [pool.submit(add, (first, second)[n % 2]) for n in range(THREADS)]
        ids = {future.result() for future in futures}

        # Assert
        assert len(ids) == 1
        assert first.collection("user:123").count() == 1


def test_two_processes_that_open_one_file_and_write_at_once_lose_nothing(path: Path) -> None:
    # Arrange: processes of their own, as two programs would be, not threads of this one. They
    # wait for each other, so that they open the missing file and write at the same time.
    spawn = multiprocessing.get_context("spawn")
    start = spawn.Barrier(2)
    contents = [[f"process {number} note {note}" for note in range(50)] for number in range(2)]
    workers = [
        spawn.Process(
            target=add_in_another_process, args=(path, "user:123", mine, start), daemon=True
        )
        for mine in contents
    ]

    # Act
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(WAIT_SECONDS)

    # Assert
    assert [worker.exitcode for worker in workers] == [0, 0]
    with MemoryEngine.sqlite(path) as engine:
        stored = sorted(record.content for record in engine.collection("user:123").list())
    assert stored == sorted(contents[0] + contents[1])
    assert run_sql(path, "PRAGMA integrity_check") == [("ok",)]
