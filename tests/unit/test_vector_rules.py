"""What happens to a record's vector when the record is written, the same in every store."""

from dataclasses import replace

import pytest

from pybodn._models import MemoryRecord
from pybodn._storage.base import (
    UNCHANGED,
    Unchanged,
    Vector,
    check_size,
    check_sizes,
    vector_after,
)
from pybodn.errors import EmbeddingError
from tests.unit.records import RECORD


def holding(text: str) -> MemoryRecord:
    return replace(RECORD, content=text)


def vector_for(text: str) -> Vector:
    return Vector("model", text, b"\x00\x00\x80\x3f")


def nothing_stored(model: str) -> int | None:
    return None


@pytest.mark.parametrize(
    ("current", "written", "vector", "after"),
    [
        pytest.param(None, "кава", vector_for("кава"), vector_for("кава"), id="new-with-vector"),
        pytest.param(None, "кава", None, None, id="new-without-vector"),
        pytest.param("кава", "кава", None, UNCHANGED, id="same-text"),
        pytest.param("кава", "чай", None, None, id="new-text-without-vector"),
        pytest.param("кава", "чай", vector_for("чай"), vector_for("чай"), id="new-text"),
        pytest.param("кава", "кава", vector_for("кава"), vector_for("кава"), id="same-text-again"),
        pytest.param("кава", "чай", vector_for("кава"), None, id="vector-of-the-old-text"),
        pytest.param("кава", "кава", vector_for("чай"), UNCHANGED, id="vector-of-another-text"),
    ],
)
def test_a_record_only_ever_has_a_vector_of_the_text_it_holds(
    current: str | None, written: str, vector: Vector | None, after: Vector | Unchanged | None
) -> None:
    # Arrange
    before = None if current is None else holding(current)

    # Act / Assert
    assert vector_after(before, holding(written), vector, nothing_stored) == after


def test_only_a_new_vector_is_checked_against_its_models_stored_vectors() -> None:
    # Arrange: the model's stored vectors have 3 numbers, and the vectors here 1
    stored = {"model": 12}.get

    # Act / Assert
    with pytest.raises(EmbeddingError, match="gave a vector of length 1, but its stored"):
        vector_after(None, holding("кава"), vector_for("кава"), stored)
    assert vector_after(holding("кава"), holding("кава"), vector_for("чай"), stored) == UNCHANGED
    assert vector_after(holding("кава"), holding("чай"), vector_for("кава"), stored) is None


@pytest.mark.parametrize("stored", [None, 4])
def test_a_vector_of_the_stored_size_or_the_first_of_its_model_is_taken(stored: int | None) -> None:
    check_size(vector_for("кава"), stored)


def test_a_vector_of_another_size_than_its_model_stored_is_refused() -> None:
    with pytest.raises(
        EmbeddingError,
        match=r"^model 'model' gave a vector of length 1, but its stored vectors have length 3:"
        " give a model of another size another name",
    ):
        check_size(vector_for("кава"), 12)


def test_vectors_written_together_agree_with_the_stored_ones_and_each_other() -> None:
    # Arrange: "a" has vectors of 4 bytes stored, "b" none yet
    stored = {"a": 4}
    vectors = [
        Vector("a", "кава", bytes(4)),
        Vector("b", "кава", bytes(8)),
        Vector("b", "чай", bytes(8)),
    ]

    # Act / Assert
    check_sizes(vectors, stored.get)
    with pytest.raises(EmbeddingError, match="'b' gave a vector of length 3, but its stored"):
        check_sizes([*vectors, Vector("b", "молоко", bytes(12))], stored.get)
