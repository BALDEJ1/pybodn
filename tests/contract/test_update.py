from dataclasses import replace

import pytest

from pybodn import Collection
from pybodn.errors import ConcurrentUpdateError, RecordNotFoundError

MISSING_ID = "0" * 32


def test_update_changes_only_the_fields_passed(notes: Collection) -> None:
    # Arrange
    original = notes.add(
        "Prefers Python",
        key="language",
        metadata={"topic": "lang"},
        tags=["pref"],
        importance=0.8,
        source="chat",
    )

    # Act
    updated = notes.update(original.id, content="Prefers Rust")

    # Assert
    expected = replace(original, content="Prefers Rust", updated_at=updated.updated_at, version=2)
    assert updated == expected
    assert notes.get(original.id) == updated


def test_update_can_change_every_field_but_the_key(notes: Collection) -> None:
    # Arrange
    original = notes.add("Prefers Python", key="language")

    # Act
    updated = notes.update(
        original.id,
        content="Prefers Rust",
        metadata={"topic": "lang"},
        tags=["pref"],
        importance=1,
        source="chat",
    )

    # Assert
    assert (updated.content, updated.metadata, updated.tags) == (
        "Prefers Rust",
        {"topic": "lang"},
        ("pref",),
    )
    assert (updated.importance, updated.source, updated.key) == (1.0, "chat", "language")


def test_update_keeps_the_creation_time_and_moves_the_update_time(notes: Collection) -> None:
    # Arrange
    original = notes.add("Prefers Python")

    # Act
    updated = notes.update(original.id, content="Prefers Rust")

    # Assert
    assert updated.created_at == original.created_at
    assert updated.updated_at > original.updated_at


def test_update_with_the_same_values_still_counts_as_a_change(notes: Collection) -> None:
    # Arrange
    original = notes.add("Prefers Python")

    # Act
    updated = notes.update(original.id, content="Prefers Python")

    # Assert
    assert updated.version == 2
    assert updated.updated_at > original.updated_at


def test_update_with_none_clears_the_source(notes: Collection) -> None:
    original = notes.add("Prefers Python", source="chat")
    assert notes.update(original.id, source=None).source is None


def test_update_keeps_the_position_in_the_list(notes: Collection) -> None:
    # Arrange
    first, middle, last = notes.add("first"), notes.add("middle"), notes.add("last")

    # Act
    updated = notes.update(middle.id, content="changed")

    # Assert
    assert notes.list() == [first, updated, last]


def test_update_with_the_current_expected_version_succeeds(notes: Collection) -> None:
    original = notes.add("Prefers Python")
    updated = notes.update(original.id, content="Prefers Rust", expected_version=original.version)
    assert updated.version == 2


def test_update_with_a_stale_expected_version_changes_nothing(notes: Collection) -> None:
    # Arrange
    original = notes.add("Prefers Python")
    current = notes.update(original.id, content="Prefers Rust")

    # Act / Assert
    with pytest.raises(ConcurrentUpdateError):
        notes.update(original.id, content="Prefers Go", expected_version=original.version)
    assert notes.get(original.id) == current


def test_a_failed_update_does_not_block_the_next_one(notes: Collection) -> None:
    # Arrange
    original = notes.add("Prefers Python")
    with pytest.raises(ConcurrentUpdateError):
        notes.update(original.id, content="Prefers Go", expected_version=2)

    # Act
    updated = notes.update(original.id, content="Prefers Rust")

    # Assert
    assert (updated.content, updated.version) == ("Prefers Rust", 2)


def test_update_of_an_unknown_id_raises_and_creates_nothing(notes: Collection) -> None:
    with pytest.raises(RecordNotFoundError):
        notes.update(MISSING_ID, content="Prefers Rust")
    assert notes.count() == 0


def test_update_of_an_unknown_id_raises_even_with_an_expected_version(notes: Collection) -> None:
    with pytest.raises(RecordNotFoundError):
        notes.update(MISSING_ID, content="Prefers Rust", expected_version=1)


def test_update_cannot_change_the_key(notes: Collection) -> None:
    # Arrange
    original = notes.add("Prefers Python", key="language")

    # Act / Assert
    with pytest.raises(TypeError, match="key"):
        notes.update(original.id, key="other")  # type: ignore[call-arg]
    assert notes.get(original.id) == original
