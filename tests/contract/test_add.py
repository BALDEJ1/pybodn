import re

import pytest

from pybodn import Collection, NewMemory
from pybodn.errors import DuplicateKeyError, ValidationError

HEX_ID = re.compile(r"[0-9a-f]{32}")


def test_add_returns_a_record_with_the_defaults(notes: Collection) -> None:
    # Act
    record = notes.add("Prefers Python")

    # Assert
    assert HEX_ID.fullmatch(record.id)
    assert record.namespace == notes.namespace == "user:123"
    assert record.content == "Prefers Python"
    assert record.key is None
    assert record.metadata == {}
    assert record.tags == ()
    assert record.importance == 0.5
    assert record.source is None
    assert record.version == 1


def test_add_keeps_every_field_passed(notes: Collection) -> None:
    # Act
    record = notes.add(
        "Prefers Python",
        key="language",
        metadata={"topic": "lang"},
        tags=["pref", "lang"],
        importance=0.8,
        source="chat",
    )

    # Assert
    assert record.key == "language"
    assert record.metadata == {"topic": "lang"}
    assert record.tags == ("pref", "lang")
    assert record.importance == 0.8
    assert record.source == "chat"


def test_an_added_record_reads_back_equal(notes: Collection) -> None:
    record = notes.add("Prefers Python", key="language", metadata={"topic": "lang"}, tags=["pref"])
    assert notes.get(record.id) == record


def test_a_new_record_has_equal_creation_and_update_times(notes: Collection) -> None:
    record = notes.add("Prefers Python")
    assert record.created_at == record.updated_at


def test_every_record_gets_its_own_id(notes: Collection) -> None:
    assert notes.add("same").id != notes.add("same").id


def test_add_many_returns_the_records_in_the_order_given(notes: Collection) -> None:
    # Act
    records = notes.add_many([NewMemory("a"), NewMemory("b", key="k")])

    # Assert
    assert [record.content for record in records] == ["a", "b"]
    assert notes.list() == records


def test_add_many_takes_any_iterable(notes: Collection) -> None:
    records = notes.add_many(NewMemory(text) for text in ("a", "b"))
    assert [record.content for record in records] == ["a", "b"]


def test_add_many_of_nothing_adds_nothing(notes: Collection) -> None:
    assert notes.add_many([]) == []
    assert notes.count() == 0


def test_records_without_keys_never_conflict(notes: Collection) -> None:
    notes.add_many([NewMemory("same"), NewMemory("same")])
    notes.add("same")
    assert notes.count() == 3


def test_a_used_key_is_rejected_and_the_original_stays(notes: Collection) -> None:
    # Arrange
    original = notes.add("first", key="language")

    # Act / Assert
    with pytest.raises(DuplicateKeyError, match="language"):
        notes.add("second", key="language")
    assert notes.list() == [original]


def test_a_key_repeated_within_a_batch_writes_nothing(notes: Collection) -> None:
    with pytest.raises(DuplicateKeyError, match="language"):
        notes.add_many([NewMemory("a", key="language"), NewMemory("b", key="language")])
    assert notes.count() == 0


def test_a_batch_with_a_used_key_writes_nothing(notes: Collection) -> None:
    # Arrange
    original = notes.add("first", key="language")

    # Act / Assert
    with pytest.raises(DuplicateKeyError, match="language"):
        notes.add_many([NewMemory("new"), NewMemory("second", key="language")])
    assert notes.list() == [original]


def test_a_batch_with_a_bad_second_memory_writes_nothing(notes: Collection) -> None:
    with pytest.raises(ValidationError, match="content"):
        notes.add_many([NewMemory("good"), NewMemory("")])
    assert notes.count() == 0
