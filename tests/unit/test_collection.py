from dataclasses import fields

import pytest

from pybodn import MemoryEngine, NewMemory
from pybodn._collection import _FIELD_CHECKS, MemoryChanges
from pybodn.errors import ValidationError


def test_every_field_a_caller_sets_has_exactly_one_check() -> None:
    assert set(_FIELD_CHECKS) == {field.name for field in fields(NewMemory)}
    assert MemoryChanges.__optional_keys__ <= set(_FIELD_CHECKS)


def test_update_checks_fields_in_one_order_whatever_order_they_are_passed_in() -> None:
    with MemoryEngine.in_memory() as engine:
        # Arrange
        notes = engine.collection("user:123")
        record = notes.add("Prefers Python")

        # Act / Assert
        with pytest.raises(ValidationError, match="content"):
            notes.update(record.id, importance=5, content="")
