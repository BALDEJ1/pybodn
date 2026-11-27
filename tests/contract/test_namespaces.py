"""``MemoryEngine.namespaces()``: every namespace that holds a record, whatever its state."""

from pybodn import MemoryEngine
from tests.contract.conftest import START


def test_an_empty_store_has_no_namespaces(engine: MemoryEngine) -> None:
    assert engine.namespaces() == []


def test_each_namespace_with_records_is_named_once_in_order(engine: MemoryEngine) -> None:
    # Arrange
    for namespace in ("user:bob", "agent:planner", "user:bob", "user:alice"):
        engine.collection(namespace).add("A note")
    engine.collection("support-bot")  # a collection alone holds no records

    # Act / Assert
    assert engine.namespaces() == ["agent:planner", "user:alice", "user:bob"]


def test_names_come_in_the_order_of_their_code_points(engine: MemoryEngine) -> None:
    # Arrange
    names = ["zeta", "Äpfel", "agent", "Zulu", "ёж", "日本", "a b"]
    for name in names:
        engine.collection(name).add("A note")

    # Act / Assert
    assert engine.namespaces() == sorted(names)


def test_a_namespace_of_hidden_records_is_still_named(engine: MemoryEngine) -> None:
    # Arrange
    archived = engine.collection("archived-only")
    archived.archive(archived.add("Planning a trip to Lisbon").id)
    # The engine's clock is already past START, so the record has expired as it is added.
    engine.collection("expired-only").add("Pickup at the south entrance", expires_at=START)

    # Act / Assert
    assert engine.namespaces() == ["archived-only", "expired-only"]


def test_a_namespace_goes_with_its_last_record(engine: MemoryEngine) -> None:
    # Arrange
    notes = engine.collection("user:123")
    first = notes.add("Prefers short answers")
    second = notes.add("Works from home on Fridays")
    engine.collection("user:456").add("Prefers metric units")

    # Act / Assert
    notes.delete(first.id)
    assert engine.namespaces() == ["user:123", "user:456"]
    notes.delete(second.id)
    assert engine.namespaces() == ["user:456"]


def test_purging_the_last_expired_records_removes_their_namespace(engine: MemoryEngine) -> None:
    # Arrange
    engine.collection("user:123").add("Pickup at the south entrance", expires_at=START)
    engine.collection("user:456").add("Prefers metric units")

    # Act
    engine.purge_expired()

    # Assert
    assert engine.namespaces() == ["user:456"]
