"""Turning a model's answer into stored vectors, and finding the vectors closest to a query."""

import math
from collections.abc import Callable, Iterator
from typing import Any

import pytest

np = pytest.importorskip("numpy")

from pybodn import Embedder, _vectors  # noqa: E402 - _vectors needs NumPy, checked above
from pybodn._storage.base import Vector  # noqa: E402
from pybodn.errors import EmbeddingError, StorageError  # noqa: E402

MODEL = "test-model"


def answering(answer: object) -> Callable[[list[str]], object]:
    """A model that gives ``answer`` to any texts."""
    return lambda texts: answer


def unit(*numbers: float) -> bytes:
    """``numbers`` scaled to unit length, as stored: 4-byte floats in little-endian order."""
    array = np.array(numbers, dtype=np.float64)
    return bytes((array / np.linalg.norm(array)).astype("<f4").tobytes())


def embed(answer: object, texts: list[str]) -> list[Vector]:
    return _vectors.embed(Embedder(MODEL, answering(answer)), texts)  # type: ignore[arg-type]


# embed -----------------------------------------------------------------------------------------


def test_each_text_gets_its_vector_at_unit_length_under_the_models_name() -> None:
    assert embed([[3, 4], [0, 2]], ["a", "b"]) == [
        Vector(MODEL, "a", unit(3, 4)),
        Vector(MODEL, "b", unit(0, 1)),
    ]


def test_texts_reach_the_model_batch_size_at_a_time_in_order() -> None:
    # Arrange
    calls: list[list[str]] = []

    def model(texts: list[str]) -> list[list[float]]:
        calls.append(texts)
        return [[1.0] for _ in texts]

    # Act
    _vectors.embed(Embedder(MODEL, model, batch_size=2), ["a", "b", "c", "d", "e"])

    # Assert
    assert calls == [["a", "b"], ["c", "d"], ["e"]]


def test_no_texts_never_call_the_model() -> None:
    def model(texts: list[str]) -> list[list[float]]:
        pytest.fail("the model must not be called")

    assert _vectors.embed(Embedder(MODEL, model), []) == []


@pytest.mark.parametrize(
    "make_answer",
    [
        pytest.param(lambda: [[3, 4]], id="integers"),
        pytest.param(lambda: [(3.0, 4.0)], id="tuple"),
        pytest.param(lambda: iter([[3.0, 4.0]]), id="iterator"),
        pytest.param(lambda: np.array([[3, 4]], dtype=np.float32), id="numpy-array"),
        pytest.param(lambda: [np.array([3, 4], dtype=np.int8)], id="numpy-integers"),
        pytest.param(lambda: [[3e100, 4e100]], id="large"),
        pytest.param(lambda: [[3e-100, 4e-100]], id="small"),
    ],
)
def test_a_model_may_answer_with_any_lists_of_real_numbers(
    make_answer: Callable[[], object],
) -> None:
    assert embed(make_answer(), ["a"]) == [Vector(MODEL, "a", unit(3, 4))]


@pytest.mark.parametrize(
    ("answer", "message"),
    [
        pytest.param(None, "must answer with a list of vectors", id="not-a-list"),
        pytest.param([], "one vector per text: it gave 0 for 1", id="too-few"),
        pytest.param([[1.0], [1.0]], "one vector per text: it gave 2 for 1", id="too-many"),
        pytest.param([1.0, 2.0], "one vector per text: it gave 2 for 1", id="one-vector-unwrapped"),
        pytest.param([1.0], "a non-empty list of numbers", id="a-number-for-a-vector"),
        pytest.param([[]], "a non-empty list of numbers", id="empty"),
        pytest.param([[[1.0], [2.0]]], "a non-empty list of numbers", id="nested"),
        pytest.param([["1"]], "a non-empty list of numbers", id="text"),
        pytest.param([[None]], "a non-empty list of numbers", id="none"),
        pytest.param([[True, False]], "a non-empty list of numbers", id="booleans"),
        pytest.param([[1j]], "a non-empty list of numbers", id="complex"),
        pytest.param(["12"], "a non-empty list of numbers", id="text-for-a-vector"),
        pytest.param([b"\x00\x00\x80?"], "a non-empty list of numbers", id="bytes"),
        pytest.param([bytearray(b"\x00\x00\x80?")], "a non-empty list of numbers", id="bytearray"),
        pytest.param(
            [memoryview(b"\x00\x00\x80?")], "a non-empty list of numbers", id="memoryview"
        ),
        pytest.param([{1.0, 2.0}], "a non-empty list of numbers", id="set"),
        pytest.param([{1.0: 2.0}], "a non-empty list of numbers", id="dict"),
        pytest.param([[10**400]], "a non-empty list of numbers", id="integer-beyond-floats"),
        pytest.param([[[1.0], [2.0, 3.0]]], "a non-empty list of numbers", id="ragged"),
        pytest.param([[0.0, 0.0]], "cannot be scaled to unit length", id="zeros"),
        pytest.param([[math.inf, 1.0]], "cannot be scaled to unit length", id="infinite"),
        pytest.param([[math.nan, 1.0]], "cannot be scaled to unit length", id="not-a-number"),
        pytest.param([[1e200, 1e200]], "cannot be scaled to unit length", id="too-large"),
        pytest.param([[1e-200, 1e-200]], "cannot be scaled to unit length", id="too-small"),
    ],
)
def test_an_unusable_answer_raises_embedding_error_naming_the_model(
    answer: object, message: str
) -> None:
    with pytest.raises(EmbeddingError, match=f"^model 'test-model' .*{message}"):
        embed(answer, ["a"])


def test_vectors_of_different_lengths_in_one_answer_are_refused() -> None:
    with pytest.raises(EmbeddingError, match="vectors of different lengths in one answer: 1, 2"):
        embed([[1.0], [1.0, 2.0]], ["a", "b"])


def failing_while_read(error: Exception) -> Iterator[Any]:
    """A lazy answer whose reading fails, as a model's may when its network does."""
    yield from ()
    raise error


@pytest.mark.parametrize("error", [TypeError("bad token"), ValueError("network down")])
def test_an_error_while_a_lazy_answer_is_read_passes_through_unchanged(error: Exception) -> None:
    for answer in (failing_while_read(error), [failing_while_read(error)]):
        with pytest.raises(type(error), match=str(error)):
            embed(answer, ["a"])


def test_an_error_of_the_model_itself_passes_through_unchanged() -> None:
    # Arrange
    def model(texts: list[str]) -> list[list[float]]:
        raise ConnectionError("no network")

    # Act / Assert
    with pytest.raises(ConnectionError, match="no network"):
        _vectors.embed(Embedder(MODEL, model), ["a"])


# best ------------------------------------------------------------------------------------------


def query(*numbers: float) -> Vector:
    return Vector(MODEL, "query", unit(*numbers))


def candidates(*vectors: tuple[float, ...]) -> list[tuple[int, bytes]]:
    """Each vector stored, under its place in the list."""
    return [(place, unit(*numbers)) for place, numbers in enumerate(vectors)]


def test_the_closest_come_first_scored_by_similarity_from_minus_one_to_one() -> None:
    # Act
    found = _vectors.best(query(1, 0), candidates((0, 1), (-1, 0), (1, 1), (1, 0)), limit=10)

    # Assert
    assert [place for _, place in found] == [3, 2, 0, 1]
    assert [score for score, _ in found] == pytest.approx([1.0, math.sqrt(0.5), 0.0, -1.0])
    assert all(type(score) is float for score, _ in found)


def test_at_most_limit_are_returned() -> None:
    found = _vectors.best(query(1, 0), candidates((0, 1), (1, 0), (1, 1)), limit=2)

    assert [place for _, place in found] == [1, 2]


def test_no_candidates_give_nothing() -> None:
    assert _vectors.best(query(1, 0), [], limit=10) == []


@pytest.mark.parametrize(("equal", "batch"), [(5, 2), (300, 64)])
def test_equally_close_candidates_keep_the_order_given_across_batches(
    monkeypatch: pytest.MonkeyPatch, equal: int, batch: int
) -> None:
    # Arrange: the equal ones arrive in several batches, and in numbers that a sort which does
    # not keep the order of equals would mix up
    monkeypatch.setattr(_vectors, "READ_BATCH", batch)

    # Act
    found = _vectors.best(query(1, 0), candidates(*[(1.0, 1.0)] * equal), limit=equal - 1)

    # Assert
    assert [place for _, place in found] == list(range(equal - 1))


def test_a_score_never_passes_one_or_minus_one() -> None:
    # Arrange: vectors that rounding to 4-byte numbers leaves a hair longer or shorter than 1
    vectors = np.random.default_rng(2026).normal(size=(500, 16))

    for numbers in vectors:
        # Act
        found = _vectors.best(query(*numbers), candidates(numbers, -numbers), limit=2)

        # Assert
        assert all(-1.0 <= score <= 1.0 for score, _ in found)


def test_reading_in_batches_finds_what_one_batch_finds(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    numbers = np.random.default_rng(2026).normal(size=(50, 8))
    stored = candidates(*(tuple(row) for row in numbers))
    target = query(*numbers[0] + numbers[1])
    at_once = _vectors.best(target, stored, limit=7)

    # Act
    monkeypatch.setattr(_vectors, "READ_BATCH", 3)
    in_batches = _vectors.best(target, stored, limit=7)

    # Assert
    assert [place for _, place in in_batches] == [place for _, place in at_once]
    assert [score for score, _ in in_batches] == pytest.approx([score for score, _ in at_once])


@pytest.mark.parametrize(
    "damaged",
    [
        pytest.param(b"\x00", id="too-short"),
        pytest.param(unit(1, 0, 0), id="too-long"),
        pytest.param("text", id="not-bytes"),
        pytest.param(None, id="missing"),
    ],
)
def test_a_stored_vector_of_another_size_is_damaged(damaged: object) -> None:
    with pytest.raises(StorageError, match="damaged"):
        _vectors.best(query(1, 0), [(0, unit(1, 0)), (1, damaged)], limit=10)
