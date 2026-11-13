import pytest

from pybodn import Collection, MemoryEngine, NewMemory
from pybodn.errors import RecordNotFoundError

MAX_INTEGER = 2**63 - 1
MISSING_ID = "0" * 32


def test_get_of_an_unknown_id_returns_none(notes: Collection) -> None:
    assert notes.get(MISSING_ID) is None


def test_get_by_key_returns_the_record_with_that_key(notes: Collection) -> None:
    # Arrange
    notes.add("Prefers dark mode", key="theme")
    record = notes.add("Python", key="language")

    # Act / Assert
    assert notes.get_by_key("language") == record


def test_get_by_key_of_an_unknown_key_returns_none(notes: Collection) -> None:
    assert notes.get_by_key("language") is None


def test_list_returns_every_record_in_the_order_added(notes: Collection) -> None:
    # Arrange
    first = notes.add("a")
    batch = notes.add_many([NewMemory("b"), NewMemory("c")])
    last = notes.add("d")

    # Act / Assert
    assert notes.list() == [first, *batch, last]


def test_list_of_an_empty_collection_is_empty(notes: Collection) -> None:
    assert notes.list() == []


@pytest.mark.parametrize(
    ("limit", "offset", "expected"),
    [
        pytest.param(2, 0, "ab", id="first-page"),
        pytest.param(2, 2, "cd", id="second-page"),
        pytest.param(2, 3, "d", id="short-last-page"),
        pytest.param(None, 1, "bcd", id="offset-without-limit"),
        pytest.param(1, 4, "", id="past-the-end"),
        pytest.param(MAX_INTEGER, 0, "abcd", id="largest-limit"),
        pytest.param(MAX_INTEGER, MAX_INTEGER, "", id="largest-limit-and-offset"),
    ],
)
def test_list_returns_one_page(
    notes: Collection, limit: int | None, offset: int, expected: str
) -> None:
    # Arrange
    notes.add_many(NewMemory(text) for text in "abcd")

    # Act
    page = notes.list(limit=limit, offset=offset)

    # Assert
    assert "".join(record.content for record in page) == expected


def test_count_counts_the_records(notes: Collection) -> None:
    notes.add_many(NewMemory(text) for text in "abc")
    assert notes.count() == 3


def test_count_of_an_empty_collection_is_zero(notes: Collection) -> None:
    assert notes.count() == 0


def test_a_record_is_invisible_from_another_namespace(engine: MemoryEngine) -> None:
    # Arrange
    mine = engine.collection("user:1")
    theirs = engine.collection("user:2")
    record = mine.add("secret", key="language")

    # Act / Assert
    assert theirs.get(record.id) is None
    assert theirs.get_by_key("language") is None
    assert theirs.list() == []
    assert theirs.count() == 0
    assert theirs.delete(record.id) is False
    with pytest.raises(RecordNotFoundError):
        theirs.update(record.id, content="changed")
    assert mine.list() == [record]


def test_one_key_in_two_namespaces_is_two_records(engine: MemoryEngine) -> None:
    # Arrange
    mine = engine.collection("user:1")
    theirs = engine.collection("user:2")

    # Act
    first = mine.put("language", "Python")
    second = theirs.put("language", "Rust")

    # Assert
    assert first.id != second.id
    assert mine.get_by_key("language") == first
    assert theirs.get_by_key("language") == second


def test_collections_of_one_namespace_share_records(engine: MemoryEngine) -> None:
    record = engine.collection("user:1").add("Prefers Python")
    assert engine.collection("user:1").get(record.id) == record
