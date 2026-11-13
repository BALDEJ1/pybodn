"""Two engines on one file stand in for two processes: each sees the other's writes at once."""

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from pybodn import MemoryEngine
from pybodn.errors import ConcurrentUpdateError, RecordNotFoundError
from tests.sqlite.support import SHORT_WAIT, THREADS, WAIT_SECONDS

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
