from dataclasses import FrozenInstanceError
from datetime import UTC, datetime

import pytest

from pybodn._models import MemoryRecord, NewMemory

NOW = datetime(2026, 11, 13, 9, 55, tzinfo=UTC)

RECORD = MemoryRecord(
    id="0" * 32,
    namespace="user:123",
    content="Prefers Python",
    key=None,
    metadata={},
    tags=(),
    importance=0.5,
    source=None,
    created_at=NOW,
    updated_at=NOW,
    version=1,
)


def test_new_memory_has_the_documented_defaults() -> None:
    # Act
    memory = NewMemory("Prefers Python")

    # Assert
    assert memory.key is None
    assert memory.metadata == {}
    assert memory.tags == ()
    assert memory.importance == 0.5
    assert memory.source is None


def test_new_memory_takes_only_content_positionally() -> None:
    with pytest.raises(TypeError):
        NewMemory("Prefers Python", "language")  # type: ignore[call-arg]


def test_memory_record_takes_only_keyword_arguments() -> None:
    with pytest.raises(TypeError):
        MemoryRecord("0" * 32)  # type: ignore[call-arg]


@pytest.mark.parametrize("model", [NewMemory("Prefers Python"), RECORD])
def test_models_are_frozen(model: NewMemory | MemoryRecord) -> None:
    with pytest.raises(FrozenInstanceError):
        model.content = "changed"  # type: ignore[misc]


@pytest.mark.parametrize("model", [NewMemory("Prefers Python"), RECORD])
def test_models_use_slots(model: NewMemory | MemoryRecord) -> None:
    assert not hasattr(model, "__dict__")
