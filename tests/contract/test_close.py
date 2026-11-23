from collections.abc import Callable
from datetime import UTC, datetime

import pytest

from pybodn import Collection, MemoryEngine, MemoryRecord, NewMemory
from pybodn._filter import EVERYTHING
from pybodn._search import Term
from pybodn._storage.base import Store, Vector
from pybodn.errors import EngineClosedError

MISSING_ID = "0" * 32
QUERY = Vector("fake-1", "Python", b"\x00\x00\x80\x3f")

ACTIONS: dict[str, Callable[[Collection], object]] = {
    "add": lambda notes: notes.add("Prefers Python"),
    "add_many": lambda notes: notes.add_many([NewMemory("Prefers Python")]),
    "add_many-of-nothing": lambda notes: notes.add_many([]),
    "add-with-dedupe": lambda notes: notes.add("Prefers Python", dedupe=True),
    "add_many-of-nothing-with-dedupe": lambda notes: notes.add_many([], dedupe=True),
    "put": lambda notes: notes.put("language", "Python"),
    "get": lambda notes: notes.get(MISSING_ID),
    "get_by_key": lambda notes: notes.get_by_key("language"),
    "update": lambda notes: notes.update(MISSING_ID, content="Prefers Rust"),
    "delete": lambda notes: notes.delete(MISSING_ID),
    "archive": lambda notes: notes.archive(MISSING_ID),
    "restore": lambda notes: notes.restore(MISSING_ID),
    "list": lambda notes: notes.list(),
    "count": lambda notes: notes.count(),
    "search": lambda notes: notes.search("Python"),
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
    "archive-with-bad-id": lambda notes: notes.archive(""),
    "get-with-bad-option": lambda notes: notes.get(MISSING_ID, include_archived=1),  # type: ignore[arg-type]
    "list-with-bad-limit": lambda notes: notes.list(limit=0),
    "list-with-bad-filter": lambda notes: notes.list(min_importance=2),
    "count-with-bad-filter": lambda notes: notes.count(tags="health"),
    "count-with-unknown-filter": lambda notes: notes.count(colour="red"),  # type: ignore[call-arg]
    "search-without-words": lambda notes: notes.search("?!"),
    "search-with-bad-mode": lambda notes: notes.search("Python", mode="nope"),  # type: ignore[arg-type]
    "search-by-meaning": lambda notes: notes.search("Python", mode="vector"),
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
    "insert": lambda store: store.insert("user:123", []),
    "insert-with-dedupe": lambda store: store.insert("user:123", [], dedupe=EVERYTHING),
    "find_copies": lambda store: store.find_copies("user:123", [], where=EVERYTHING),
    "modify": lambda store: store.modify("user:123", "id", MISSING_ID, never_build),
    "delete": lambda store: store.delete("user:123", MISSING_ID),
    "purge_expired": lambda store: store.purge_expired(datetime.now(UTC), limit=10),
    "erase_deleted": lambda store: store.erase_deleted(fast=False),
    "list": lambda store: store.list("user:123", where=EVERYTHING, limit=None, offset=0),
    "count": lambda store: store.count("user:123", where=EVERYTHING),
    "search_words": lambda store: store.search_words(
        "user:123", (Term("python", is_prefix=False),), where=EVERYTHING, limit=10
    ),
    "search_vectors": lambda store: store.search_vectors(
        "user:123", QUERY, where=EVERYTHING, limit=10
    ),
    "last_seq": lambda store: store.last_seq(),
    "missing_vectors": lambda store: store.missing_vectors("fake-1", after=0, until=1, limit=10),
    "write_vectors": lambda store: store.write_vectors([]),
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


@pytest.mark.parametrize("name", ["embed_missing", "purge_expired", "erase_deleted"])
def test_an_engine_wide_call_after_close_raises_engine_closed_error(
    engine: MemoryEngine, name: str
) -> None:
    # Closed is reported first, before a missing model too.
    engine.close()
    with pytest.raises(EngineClosedError):
        getattr(engine, name)()


def test_a_closed_engine_is_reported_before_a_bad_fast_flag(engine: MemoryEngine) -> None:
    engine.close()
    with pytest.raises(EngineClosedError):
        engine.erase_deleted(fast=1)  # type: ignore[arg-type]


def test_close_can_be_called_again(engine: MemoryEngine) -> None:
    engine.close()
    engine.close()
