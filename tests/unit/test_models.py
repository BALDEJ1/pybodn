from dataclasses import FrozenInstanceError

import pytest

from pybodn._models import MemoryRecord, NewMemory, SearchHit
from tests.unit.records import RECORD


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


def test_a_search_hit_takes_only_keyword_arguments() -> None:
    with pytest.raises(TypeError):
        SearchHit(RECORD, 1.5)  # type: ignore[call-arg]


def test_a_search_hit_is_frozen_and_uses_slots() -> None:
    # Arrange
    hit = SearchHit(record=RECORD, score=1.5)

    # Act / Assert
    with pytest.raises(FrozenInstanceError):
        hit.score = 2.0  # type: ignore[misc]
    assert not hasattr(hit, "__dict__")
