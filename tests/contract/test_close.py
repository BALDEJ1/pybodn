from collections.abc import Callable

import pytest

from pybodn import Collection, MemoryEngine, NewMemory
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


def test_a_collection_cannot_be_taken_after_close(engine: MemoryEngine) -> None:
    engine.close()
    with pytest.raises(EngineClosedError):
        engine.collection("user:123")


def test_a_closed_engine_cannot_be_entered_again(engine: MemoryEngine) -> None:
    engine.close()
    with pytest.raises(EngineClosedError), engine:
        pass


def test_close_can_be_called_again(engine: MemoryEngine) -> None:
    engine.close()
    engine.close()
