"""Every operation checks its input, and a rejected write changes nothing."""

import math
from collections.abc import Callable
from typing import Any

import pytest

from pybodn import Collection, MemoryEngine, NewMemory
from pybodn.errors import ValidationError

MAX_INTEGER = 2**63 - 1
LONE_SURROGATE = "\ud800"


def nested(levels: int) -> dict[str, object]:
    """A dict holding lists, ``levels`` containers deep in total."""
    value: object = "leaf"
    for _ in range(levels - 1):
        value = [value]
    return {"deep": value}


BAD_FIELDS = [
    pytest.param({"importance": math.nan}, id="nan-importance"),
    pytest.param({"importance": True}, id="true-importance"),
    pytest.param({"importance": 1.5}, id="importance-above-one"),
    pytest.param({"metadata": {"value": math.inf}}, id="infinity-in-metadata"),
    pytest.param({"metadata": {"value": {1, 2}}}, id="set-in-metadata"),
    pytest.param({"metadata": {1: "value"}}, id="number-key-in-metadata"),
    pytest.param({"metadata": nested(101)}, id="too-deep-metadata"),
    pytest.param({"metadata": {"value": 10**5000}}, id="huge-integer-in-metadata"),
    pytest.param({"metadata": {"value": LONE_SURROGATE}}, id="lone-surrogate-in-metadata"),
    pytest.param({"tags": [""]}, id="empty-tag"),
    pytest.param({"tags": "pref"}, id="tags-as-one-string"),
    pytest.param({"source": LONE_SURROGATE}, id="lone-surrogate-source"),
]

# Rejected wherever a non-empty string is required: content, key, id, namespace.
BAD_TEXT = [
    pytest.param("", id="empty"),
    pytest.param(LONE_SURROGATE, id="lone-surrogate"),
    pytest.param(None, id="none"),
    pytest.param(1, id="number"),
]


@pytest.mark.parametrize("fields", BAD_FIELDS)
def test_add_rejects_bad_fields_and_writes_nothing(
    notes: Collection, fields: dict[str, Any]
) -> None:
    with pytest.raises(ValidationError):
        notes.add("Prefers Python", **fields)
    assert notes.count() == 0


@pytest.mark.parametrize("fields", BAD_FIELDS)
def test_put_rejects_bad_fields_and_keeps_the_old_record(
    notes: Collection, fields: dict[str, Any]
) -> None:
    # Arrange
    original = notes.put("language", "Python")

    # Act / Assert
    with pytest.raises(ValidationError):
        notes.put("language", "Rust", **fields)
    assert notes.list() == [original]


@pytest.mark.parametrize("fields", BAD_FIELDS)
def test_update_rejects_bad_fields_and_changes_nothing(
    notes: Collection, fields: dict[str, Any]
) -> None:
    # Arrange
    original = notes.add("Prefers Python")

    # Act / Assert
    with pytest.raises(ValidationError):
        notes.update(original.id, **fields)
    assert notes.list() == [original]


@pytest.mark.parametrize("field", ["content", "metadata", "tags", "importance"])
def test_update_rejects_none_for_a_field_that_needs_a_value(notes: Collection, field: str) -> None:
    # Arrange
    original = notes.add("Prefers Python")
    changes: dict[str, Any] = {field: None}

    # Act / Assert
    with pytest.raises(ValidationError, match=field):
        notes.update(original.id, **changes)
    assert notes.list() == [original]


def test_update_without_fields_is_rejected(notes: Collection) -> None:
    # Arrange
    original = notes.add("Prefers Python")

    # Act / Assert
    with pytest.raises(ValidationError, match="at least one field"):
        notes.update(original.id)
    with pytest.raises(ValidationError, match="at least one field"):
        notes.update(original.id, expected_version=1)
    assert notes.list() == [original]


@pytest.mark.parametrize("version", [0, -1, MAX_INTEGER + 1, True, "1", 1.0])
def test_update_rejects_a_bad_expected_version(notes: Collection, version: Any) -> None:
    # Arrange
    original = notes.add("Prefers Python")

    # Act / Assert
    with pytest.raises(ValidationError, match="expected_version"):
        notes.update(original.id, content="Prefers Rust", expected_version=version)
    assert notes.list() == [original]


@pytest.mark.parametrize("content", BAD_TEXT)
def test_bad_content_is_rejected_by_every_write(notes: Collection, content: Any) -> None:
    # Arrange
    original = notes.add("Prefers Python")

    # Act / Assert
    with pytest.raises(ValidationError, match="content"):
        notes.add(content)
    with pytest.raises(ValidationError, match="content"):
        notes.add_many([NewMemory(content)])
    with pytest.raises(ValidationError, match="content"):
        notes.put("language", content)
    with pytest.raises(ValidationError, match="content"):
        notes.update(original.id, content=content)
    assert notes.list() == [original]


@pytest.mark.parametrize("key", BAD_TEXT)
def test_bad_key_is_rejected_by_put_and_get_by_key(notes: Collection, key: Any) -> None:
    with pytest.raises(ValidationError, match="key"):
        notes.put(key, "Python")
    with pytest.raises(ValidationError, match="key"):
        notes.get_by_key(key)
    assert notes.count() == 0


@pytest.mark.parametrize("key", ["", LONE_SURROGATE, 1])
def test_bad_key_is_rejected_by_add(notes: Collection, key: Any) -> None:
    with pytest.raises(ValidationError, match="key"):
        notes.add("Python", key=key)
    assert notes.count() == 0


@pytest.mark.parametrize("record_id", BAD_TEXT)
def test_bad_id_is_rejected(notes: Collection, record_id: Any) -> None:
    with pytest.raises(ValidationError, match="id"):
        notes.get(record_id)
    with pytest.raises(ValidationError, match="id"):
        notes.update(record_id, content="Prefers Rust")
    with pytest.raises(ValidationError, match="id"):
        notes.delete(record_id)


@pytest.mark.parametrize("namespace", BAD_TEXT)
def test_bad_namespace_is_rejected(engine: MemoryEngine, namespace: Any) -> None:
    with pytest.raises(ValidationError, match="namespace"):
        engine.collection(namespace)


@pytest.mark.parametrize("memories", [None, 5, "text", [None], [{"content": "text"}]])
def test_add_many_rejects_anything_but_new_memories(notes: Collection, memories: Any) -> None:
    with pytest.raises(ValidationError, match="memories"):
        notes.add_many(memories)
    assert notes.count() == 0


@pytest.mark.parametrize("limit", [0, -1, MAX_INTEGER + 1, True, 1.0, "1"])
def test_bad_limit_is_rejected(notes: Collection, limit: Any) -> None:
    with pytest.raises(ValidationError, match="limit"):
        notes.list(limit=limit)


@pytest.mark.parametrize("offset", [-1, MAX_INTEGER + 1, None, False, 0.0])
def test_bad_offset_is_rejected(notes: Collection, offset: Any) -> None:
    with pytest.raises(ValidationError, match="offset"):
        notes.list(offset=offset)


# Like any Python function, a misspelled keyword is a TypeError that names the method called.
UNKNOWN_FIELD: dict[str, Callable[[Collection], object]] = {
    "add": lambda notes: notes.add("Python", colour="red"),  # type: ignore[call-arg]
    "put": lambda notes: notes.put("language", "Python", colour="red"),  # type: ignore[call-arg]
    "update": lambda notes: notes.update(notes.add("Python").id, colour="red"),  # type: ignore[call-arg]
}


@pytest.mark.parametrize("method", UNKNOWN_FIELD)
def test_an_unknown_field_is_a_type_error_naming_the_method(notes: Collection, method: str) -> None:
    with pytest.raises(
        TypeError, match=rf"^{method}\(\) got an unexpected keyword argument 'colour'$"
    ):
        UNKNOWN_FIELD[method](notes)
