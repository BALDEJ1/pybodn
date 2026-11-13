from pybodn import Collection, NewMemory

MISSING_ID = "0" * 32


def test_delete_removes_the_record_once(notes: Collection) -> None:
    # Arrange
    record = notes.add("Prefers Python")

    # Act / Assert
    assert notes.delete(record.id) is True
    assert notes.delete(record.id) is False
    assert notes.get(record.id) is None
    assert notes.count() == 0


def test_delete_of_an_unknown_id_returns_false(notes: Collection) -> None:
    assert notes.delete(MISSING_ID) is False


def test_delete_frees_the_key(notes: Collection) -> None:
    # Arrange
    record = notes.add("Python", key="language")

    # Act
    notes.delete(record.id)

    # Assert
    assert notes.get_by_key("language") is None
    again = notes.add("Rust", key="language")
    assert notes.get_by_key("language") == again


def test_delete_keeps_the_order_of_the_rest(notes: Collection) -> None:
    # Arrange
    first, middle, last = notes.add_many(NewMemory(text) for text in "abc")

    # Act
    notes.delete(middle.id)

    # Assert
    assert notes.list() == [first, last]
