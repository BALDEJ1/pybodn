"""What the public API cannot show about the in-memory store."""

from datetime import UTC, datetime

import pytest

from pybodn._filter import EVERYTHING
from pybodn._models import JSONValue, MemoryRecord
from pybodn._storage.memory import InMemoryStore

NOW = datetime(2026, 11, 13, 9, 55, tzinfo=UTC)


class Uncopyable:
    """Fails to copy, as deep metadata can when the call stack is already deep."""

    def __deepcopy__(self, memo: dict[int, object]) -> "Uncopyable":
        raise RecursionError("maximum recursion depth exceeded")


def record(
    record_id: str, key: str | None = None, metadata: dict[str, JSONValue] | None = None
) -> MemoryRecord:
    return MemoryRecord(
        id=record_id,
        namespace="user:123",
        content="Prefers Python",
        key=key,
        metadata={} if metadata is None else metadata,
        tags=(),
        importance=0.5,
        source=None,
        created_at=NOW,
        updated_at=NOW,
        version=1,
    )


def test_a_batch_that_fails_to_copy_writes_nothing() -> None:
    # Arrange
    store = InMemoryStore()
    batch = [
        record("a"),
        record("b", metadata={"value": Uncopyable()}),  # type: ignore[dict-item]
        record("c"),
    ]

    # Act / Assert
    with pytest.raises(RecursionError):
        store.insert(batch)
    assert store.count("user:123", where=EVERYTHING) == 0


def test_deleting_the_last_record_of_a_namespace_releases_the_namespace() -> None:
    # Arrange
    store = InMemoryStore()
    store.insert([record("a", key="language"), record("b")])

    # Act
    store.delete("user:123", "a")
    store.delete("user:123", "b")

    # Assert
    assert store._namespaces == {}


def test_deleting_a_record_releases_its_key() -> None:
    # Arrange
    store = InMemoryStore()
    store.insert([record("a", key="language"), record("b")])

    # Act
    store.delete("user:123", "a")

    # Assert
    assert store._namespaces["user:123"].ids_by_key == {}


def test_close_releases_every_record() -> None:
    # Arrange
    store = InMemoryStore()
    store.insert([record("a")])

    # Act
    store.close()

    # Assert
    assert store._namespaces == {}
