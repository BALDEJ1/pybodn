from pybodn import Collection


def test_put_creates_a_record_for_a_new_key(notes: Collection) -> None:
    # Act
    record = notes.put("language", "Python")

    # Assert
    assert (record.key, record.content, record.version) == ("language", "Python", 1)
    assert record.created_at == record.updated_at
    assert notes.get_by_key("language") == record


def test_put_replaces_every_field_and_resets_the_ones_not_passed(notes: Collection) -> None:
    # Arrange
    first = notes.put(
        "language",
        "Python",
        metadata={"topic": "lang"},
        tags=["pref"],
        importance=0.9,
        source="chat",
    )

    # Act
    second = notes.put("language", "Rust")

    # Assert
    assert (second.content, second.metadata, second.tags) == ("Rust", {}, ())
    assert (second.importance, second.source) == (0.5, None)
    assert notes.get(first.id) == second


def test_put_keeps_the_id_and_creation_time_and_counts_the_change(notes: Collection) -> None:
    # Arrange
    first = notes.put("language", "Python")

    # Act
    second = notes.put("language", "Rust")

    # Assert
    assert second.id == first.id
    assert second.created_at == first.created_at
    assert second.updated_at > first.updated_at
    assert second.version == 2


def test_put_with_the_same_values_still_counts_as_a_change(notes: Collection) -> None:
    # Arrange
    first = notes.put("language", "Python")

    # Act
    second = notes.put("language", "Python")

    # Assert
    assert second.version == 2
    assert second.updated_at > first.updated_at


def test_put_keeps_the_position_in_the_list(notes: Collection) -> None:
    # Arrange
    first = notes.add("first")
    notes.put("language", "Python")
    last = notes.add("last")

    # Act
    replaced = notes.put("language", "Rust")

    # Assert
    assert notes.list() == [first, replaced, last]


def test_put_replaces_a_record_added_with_that_key(notes: Collection) -> None:
    # Arrange
    added = notes.add("Python", key="language")

    # Act
    replaced = notes.put("language", "Rust")

    # Assert
    assert replaced.id == added.id
    assert notes.list() == [replaced]
