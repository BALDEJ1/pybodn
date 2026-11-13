"""Values come back exactly as stored, and nothing outside can change what is stored."""

from collections.abc import Callable

import pytest

from pybodn import Collection, JSONValue, MemoryEngine, MemoryRecord, NewMemory

TEXTS = [
    pytest.param("юнікод і емодзі 🙂", id="unicode"),
    pytest.param("nul\x00inside", id="nul"),
    pytest.param("x" * 100_000, id="long"),
]


@pytest.mark.parametrize("text", TEXTS)
def test_text_comes_back_unchanged_in_every_field(engine: MemoryEngine, text: str) -> None:
    # Arrange
    notes = engine.collection(text)

    # Act
    record = notes.put(text, text, metadata={text: text}, tags=[text], source=text)

    # Assert
    assert (record.namespace, record.key, record.content, record.source) == (text, text, text, text)
    assert record.metadata == {text: text}
    assert record.tags == (text,)
    assert notes.get_by_key(text) == record


def test_an_integer_importance_comes_back_as_a_float(notes: Collection) -> None:
    # Act
    record = notes.add("Prefers Python", importance=1)

    # Assert
    stored = notes.get(record.id)
    assert stored is not None
    assert type(record.importance) is float
    assert type(stored.importance) is float
    assert stored.importance == 1.0


def test_every_json_type_in_metadata_comes_back_equal(notes: Collection) -> None:
    # Arrange
    metadata: dict[str, JSONValue] = {
        "text": "text",
        "smallest": -(2**63),
        "largest": 2**63 - 1,
        "float": 2.5,
        "whole-float": 1.0,
        "true": True,
        "false": False,
        "none": None,
        "list": [1, "a", [None]],
        "dict": {"empty": {}},
    }

    # Act
    record = notes.add("Prefers Python", metadata=metadata)

    # Assert
    stored = notes.get(record.id)
    assert stored is not None
    assert stored.metadata == metadata
    assert [type(value) for value in stored.metadata.values()] == [
        type(value) for value in metadata.values()
    ]


def test_metadata_keeps_its_key_order(notes: Collection) -> None:
    # Act
    record = notes.add("Prefers Python", metadata={"b": 1, "a": {"d": 2, "c": 3}})

    # Assert
    stored = notes.get(record.id)
    assert stored is not None
    assert list(stored.metadata) == ["b", "a"]
    inner = stored.metadata["a"]
    assert isinstance(inner, dict)
    assert list(inner) == ["d", "c"]


def test_changing_what_was_passed_to_add_changes_nothing_stored(notes: Collection) -> None:
    # Arrange
    inner: list[JSONValue] = [1]
    metadata: dict[str, JSONValue] = {"list": inner}
    tags = ["pref"]
    record = notes.add("Prefers Python", metadata=metadata, tags=tags)

    # Act
    inner.append(2)
    metadata["new"] = True
    tags.append("other")

    # Assert
    assert record.metadata == {"list": [1]}
    assert record.tags == ("pref",)
    assert notes.get(record.id) == record


# Each one leaves the record "language" with metadata {"list": [1]} stored and returns it.


def from_add(notes: Collection) -> MemoryRecord | None:
    return notes.add("Python", key="language", metadata={"list": [1]})


def from_add_many(notes: Collection) -> MemoryRecord | None:
    return notes.add_many([NewMemory("Python", key="language", metadata={"list": [1]})])[0]


def from_put(notes: Collection) -> MemoryRecord | None:
    return notes.put("language", "Python", metadata={"list": [1]})


def from_get(notes: Collection) -> MemoryRecord | None:
    record = notes.add("Python", key="language", metadata={"list": [1]})
    return notes.get(record.id)


def from_get_by_key(notes: Collection) -> MemoryRecord | None:
    notes.add("Python", key="language", metadata={"list": [1]})
    return notes.get_by_key("language")


def from_list(notes: Collection) -> MemoryRecord | None:
    notes.add("Python", key="language", metadata={"list": [1]})
    return notes.list()[0]


def from_update_of_metadata(notes: Collection) -> MemoryRecord | None:
    record = notes.add("Python", key="language", metadata={"other": 0})
    return notes.update(record.id, metadata={"list": [1]})


def from_update_of_content(notes: Collection) -> MemoryRecord | None:
    record = notes.add("Python", key="language", metadata={"list": [1]})
    return notes.update(record.id, content="Rust")


@pytest.mark.parametrize(
    "returned_by",
    [
        from_add,
        from_add_many,
        from_put,
        from_get,
        from_get_by_key,
        from_list,
        from_update_of_metadata,
        from_update_of_content,
    ],
)
def test_changing_a_returned_record_changes_nothing_stored(
    notes: Collection, returned_by: Callable[[Collection], MemoryRecord | None]
) -> None:
    # Arrange
    returned = returned_by(notes)
    assert returned is not None

    # Act
    inner = returned.metadata["list"]
    assert isinstance(inner, list)
    inner.append(2)
    returned.metadata["new"] = True

    # Assert
    stored = notes.get_by_key("language")
    assert stored is not None
    assert stored.metadata == {"list": [1]}
