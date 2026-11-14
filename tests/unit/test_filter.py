from dataclasses import fields

import pytest

from pybodn import MemoryEngine
from pybodn._filter import _CHECKS, EVERYTHING, Filter, Filters, checked
from pybodn.errors import ValidationError


def test_every_filter_has_exactly_one_check_and_one_field() -> None:
    assert set(_CHECKS) == Filters.__optional_keys__
    assert set(_CHECKS) == {field.name for field in fields(Filter)}


def test_no_filters_keep_everything() -> None:
    assert checked({}) == EVERYTHING


def test_filters_are_checked_in_one_order_whatever_order_they_are_passed_in() -> None:
    with MemoryEngine.in_memory() as engine:
        # Arrange
        notes = engine.collection("user:123")

        # Act / Assert
        with pytest.raises(ValidationError, match="tag"):
            notes.list(created_since=None, tags=None)  # type: ignore[arg-type]
