"""Search by meaning, the same for every store: a model's answer made into stored vectors, and the
stored vectors closest to a query.

Needs NumPy, so it is imported only once an engine has an embedding model.
"""

import itertools
from collections.abc import Iterable, Iterator, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from pybodn._models import Embedder
from pybodn._storage.base import FLOAT_BYTES, Vector
from pybodn.errors import EmbeddingError, StorageError

# How many stored vectors a search scores at a time: the memory it needs stays the same however
# many records there are.
READ_BATCH = 2048

# A number of a stored vector.
_FLOAT = np.dtype(f"<f{FLOAT_BYTES}")

_NOT_NUMBERS = "must give each text a non-empty list of numbers"


def embed(embedder: Embedder, texts: Sequence[str]) -> list[Vector]:
    """The vector ``embedder``'s model gives each text, at unit length, as stored.

    The model gets ``batch_size`` texts at a time, and none without texts. An answer that is not
    one vector of real numbers per text, all as long, or a vector that cannot be scaled to unit
    length, raises ``EmbeddingError``; an error of the model itself passes through unchanged.
    """
    vectors: list[Vector] = []
    for batch in itertools.batched(texts, embedder.batch_size):
        given = list(batch)
        answer = _unit_vectors(embedder.name, embedder.embed(given), len(given))
        vectors.extend(
            Vector(embedder.name, text, data) for text, data in zip(given, answer, strict=True)
        )
    return vectors


def best[K](
    query: Vector, candidates: Iterable[tuple[K, object]], *, limit: int
) -> list[tuple[float, K]]:
    """The ``limit`` candidates closest to ``query``, closest first, each with its similarity to
    it from -1 to 1; equally close ones keep the order given.

    A candidate is a key and a stored vector of ``query``'s model. One that is not as long as
    ``query`` is damaged and raises ``StorageError``.
    """
    target = np.frombuffer(query.data, dtype=_FLOAT)
    kept_scores = np.empty(0, dtype=_FLOAT)
    kept_keys: list[K] = []
    for batch in itertools.batched(candidates, READ_BATCH):
        scores = np.concatenate((kept_scores, _similarities(target, [data for _, data in batch])))
        keys = [*kept_keys, *(key for key, _ in batch)]
        order = np.argsort(-scores, kind="stable")[:limit]
        kept_scores = scores[order]
        kept_keys = [keys[place] for place in order]
    # Rounding may take a similarity a hair past 1 or -1.
    similarities = np.clip(kept_scores, -1.0, 1.0)
    return [(float(score), key) for score, key in zip(similarities, kept_keys, strict=True)]


def _similarities(target: npt.NDArray[np.float32], stored: Sequence[object]) -> npt.NDArray[Any]:
    """The similarity of each stored vector to ``target``, a vector of the same model.

    The vectors are copied into one array, which is gone once this returns.
    """
    rows = np.frombuffer(
        bytearray().join(_stored(data, target.nbytes) for data in stored), dtype=_FLOAT
    ).reshape(len(stored), len(target))
    # Each product, then each row's sum, where a matrix product would do: a sum along a row runs
    # the same way whatever the row's place, so equal vectors get exactly equal scores.
    np.multiply(rows, target, out=rows)
    return rows.sum(axis=1)


def _unit_vectors(model: str, answer: object, texts: int) -> list[bytes]:
    """The vectors of ``model``'s answer for ``texts`` texts, each at unit length, as stored."""
    if not isinstance(answer, Iterable):
        raise _unusable(model, f"must answer with a list of vectors, not {type(answer).__name__}")
    vectors = list(answer)  # an error of the model's own iterator passes through
    if len(vectors) != texts:
        raise _unusable(
            model, f"must answer with one vector per text: it gave {len(vectors)} for {texts}"
        )
    arrays = [_numbers(model, vector) for vector in vectors]
    lengths = sorted({len(array) for array in arrays})
    if len(lengths) > 1:
        raise _unusable(
            model,
            f"gave vectors of different lengths in one answer: {', '.join(map(str, lengths))}",
        )
    return [_unit(model, array) for array in arrays]


def _numbers(model: str, vector: object) -> npt.NDArray[np.float64]:
    """The numbers of ``vector``: a non-empty list, tuple, iterator or array of real numbers.

    Anything else, such as text, bytes or a set, is refused rather than read as numbers.
    """
    values = list(vector) if isinstance(vector, Iterator) else vector  # its errors pass through
    if not isinstance(values, list | tuple) and not hasattr(values, "__array__"):
        raise _unusable(model, _NOT_NUMBERS)
    try:
        array: npt.NDArray[Any] = np.asarray(values)
    except ValueError as error:  # lists of different lengths
        raise _unusable(model, _NOT_NUMBERS) from error
    if array.ndim != 1 or array.size == 0 or array.dtype.kind not in "iuf":
        raise _unusable(model, _NOT_NUMBERS)
    return array.astype(np.float64)


def _unit(model: str, numbers: npt.NDArray[np.float64]) -> bytes:
    """``numbers`` scaled to unit length, as stored."""
    # Squares too large for a float overflow to infinity, refused below; NumPy on some systems
    # also warns of it.
    with np.errstate(over="ignore"):
        length = float(np.linalg.norm(numbers))
    # Infinite or not a number: a number was, or its squares overflowed; zero: they all were, or
    # their squares underflowed.
    if not 0 < length < float("inf"):
        raise _unusable(
            model,
            "gave a vector that cannot be scaled to unit length: its numbers are all zero,"
            " not finite, or too far from 1",
        )
    return (numbers / length).astype(_FLOAT).tobytes()


def _stored(data: object, size: int) -> bytes:
    """``data``, a stored vector, which must be ``size`` bytes long."""
    if not isinstance(data, bytes) or len(data) != size:
        raise StorageError(
            f"a vector in the database is damaged: it is not {size // FLOAT_BYTES} numbers long,"
            " as its model's are"
        )
    return data


def _unusable(model: str, problem: str) -> EmbeddingError:
    return EmbeddingError(f"model {model!r} {problem}")
