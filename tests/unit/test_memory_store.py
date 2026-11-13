"""What the public API cannot show: the in-memory store gives memory back."""

from datetime import UTC, datetime

from pybodn._models import MemoryRecord
from pybodn._storage.memory import InMemoryStore

NOW = datetime(2026, 11, 13, 9, 55, tzinfo=UTC)


def record(record_id: str, key: str | None = None) -> MemoryRecord:
    return MemoryRecord(
        id=record_id,
        namespace="user:123",
        content="Prefers Python",
        key=key,
        metadata={},
        tags=(),
        importance=0.5,
        source=None,
        created_at=NOW,
        updated_at=NOW,
        version=1,
    )


def test_deleting_the_last_record_of_a_namespace_releases_the_namespace() -> None:
    # Arrange
    store = InMemoryStore()
    store.insert([record("a", key="language"), record("b")])

    # Act
    store.delete("user:123", "a")
    store.delete("user:123", "b")

    # Assert
    assert store._namespaces == {}


def test_close_releases_every_record() -> None:
    # Arrange
    store = InMemoryStore()
    store.insert([record("a")])

    # Act
    store.close()

    # Assert
    assert store._namespaces == {}
