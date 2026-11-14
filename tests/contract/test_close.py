from collections.abc import Callable

import pytest

from pybodn import Collection, MemoryEngine, MemoryRecord, NewMemory
from pybodn._storage.base import Store
from pybodn.errors import EngineClosedError

MISSING_ID = "0" * 32

ACTIONS: dict[str, Callable[[Collection], object]] = {
    "add": lambda notes: notes.add("Prefers Python"),
    "add_many": lambda notes: notes.add_many([NewMemory("Prefers Python")]),
    "add_many-of-nothing": lambda notes: notes.add_many([]),
    "put": lambda notes: notes.put("language", "Python"),
    "get": lambda notes: notes.get(MISSING_ID),
    "get_by_key": lambda notes: notes.get_by_key("language"),
    "update": lambda notes: notes.update(MISSING_ID, content="Prefers Rust"),
    "delete": lambda notes: notes.delete(MISSING_ID),
    "list": lambda notes: notes.list(),
    "count": lambda notes: notes.count(),
    # Closed is reported before bad input, as sqlite3 does for a closed connection.
    "add-with-bad-content": lambda notes: notes.add(""),
    "add-with-unknown-field": lambda notes: notes.add("a", colour="red"),  # type: ignore[call-arg]
    "add_many-with-bad-items": lambda notes: notes.add_many([None]),  # type: ignore[list-item]
    "put-with-bad-key": lambda notes: notes.put("", "Python"),
    "get-with-bad-id": lambda notes: notes.get(""),
    "get_by_key-with-bad-key": lambda notes: notes.get_by_key(""),
    "update-without-fields": lambda notes: notes.update(MISSING_ID),
    "update-with-unknown-field": lambda notes: notes.update(MISSING_ID, key="k"),  # type: ignore[call-arg]
    "delete-with-bad-id": lambda notes: notes.delete(""),
    "list-with-bad-limit": lambda notes: notes.list(limit=0),
}


@pytest.mark.parametrize("action", ACTIONS.values(), ids=ACTIONS.keys())
def test_every_action_after_close_raises_engine_closed_error(
    engine: MemoryEngine, action: Callable[[Collection], object]
) -> None:
    # Arrange
    notes = engine.collection("user:123")
    engine.close()

    # Act / Assert
    with pytest.raises(EngineClosedError):
        action(notes)


def never_build(current: MemoryRecord | None) -> MemoryRecord:
    pytest.fail("a closed store must not build a record")


# A call that races close() gets past the engine's own check, so the store checks again.
STORE_CALLS: dict[str, Callable[[Store], object]] = {
    "get": lambda store: store.get("user:123", "id", MISSING_ID),
    "insert": lambda store: store.insert([]),
    "modify": lambda store: store.modify("user:123", "id", MISSING_ID, never_build),
    "delete": lambda store: store.delete("user:123", MISSING_ID),
    "list": lambda store: store.list("user:123", limit=None, offset=0),
    "count": lambda store: store.count("user:123"),
}


@pytest.mark.parametrize("call", STORE_CALLS.values(), ids=STORE_CALLS.keys())
def test_every_store_call_after_close_raises_engine_closed_error(
    engine: MemoryEngine, call: Callable[[Store], object]
) -> None:
    # Arrange
    store = engine._store
    engine.close()

    # Act / Assert
    with pytest.raises(EngineClosedError):
        call(store)


@pytest.mark.parametrize("namespace", ["user:123", ""])
def test_a_collection_cannot_be_taken_after_close(engine: MemoryEngine, namespace: str) -> None:
    engine.close()
    with pytest.raises(EngineClosedError):
        engine.collection(namespace)


def test_a_closed_engine_cannot_be_entered_again(engine: MemoryEngine) -> None:
    engine.close()
    with pytest.raises(EngineClosedError), engine:
        pass


def test_close_can_be_called_again(engine: MemoryEngine) -> None:
    engine.close()
    engine.close()
