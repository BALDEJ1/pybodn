"""The contract every storage implements.

A store makes no decisions: it only reads and writes, atomically. Checks, ids, times,
versions, what ``put`` and ``update`` mean, and what becomes of a record's vector live above
it or here, once for every store.

Errors: whatever ``build`` raises, and any pybodn error, passes through unchanged. A
failure of the storage itself becomes ``StorageError`` with the original as its cause.
Any call after ``close()`` raises ``EngineClosedError``.
"""

import builtins
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from types import MappingProxyType
from typing import Literal, Protocol

from pybodn._filter import Filter
from pybodn._models import MemoryRecord, SearchHit
from pybodn._search import Term
from pybodn.errors import DuplicateKeyError, EmbeddingError, EngineClosedError

# The field that ``value`` is matched against.
type Lookup = Literal["id", "key"]

# Turns the current record, or ``None`` if there is none, into the record to write.
type Build = Callable[[MemoryRecord | None], MemoryRecord]

# What a record shares with its copies: its text and its key, or ``None`` for none.
type TextAndKey = tuple[str, str | None]

# Each number of a stored vector is a float of this many bytes, in little-endian order.
FLOAT_BYTES = 4


@dataclass(frozen=True, slots=True)
class Vector:
    """What ``model`` made of ``text``: ``data`` is the vector at unit length, as stored."""

    model: str
    text: str
    data: bytes


@dataclass(frozen=True, slots=True)
class MissingVector:
    """A record that has no vector of a model: its place in the order added, and its text."""

    seq: int
    namespace: str
    id: str
    text: str


class Unchanged(Enum):
    """A write that leaves the record's vector as it is."""

    VECTOR = "the record's vector stays"


UNCHANGED = Unchanged.VECTOR

# No vectors to write along with records.
NO_VECTORS: Mapping[str, Vector] = MappingProxyType({})


class Store(Protocol):
    """Atomic reads and writes of records, kept apart by namespace."""

    @property
    def closed(self) -> bool:
        """Whether ``close()`` has been called."""

    def get(self, namespace: str, by: Lookup, value: str) -> MemoryRecord | None:
        """The record of ``namespace`` whose ``by`` equals ``value``, or ``None``."""

    def find_copies(
        self, namespace: str, records: Sequence[MemoryRecord], *, where: Filter
    ) -> dict[TextAndKey, MemoryRecord]:
        """For each text and key of ``records``, the first record added to ``namespace`` that
        ``where`` keeps and holds them, if there is one."""

    def insert(
        self,
        namespace: str,
        records: Sequence[MemoryRecord],
        *,
        vectors: Mapping[str, Vector] = NO_VECTORS,
        dedupe: Filter | None = None,
    ) -> builtins.list[MemoryRecord]:
        """Write the records of ``namespace``, each with the vector ``vectors`` holds under its
        id, or none of them, and return a record for each, in the same order.

        Without ``dedupe``, every record is written and returned. With it, a record that has a
        copy, as ``split_batch`` finds it among the records ``dedupe`` keeps, is not written, and
        its copy is returned in its place. A key that the records written repeat among
        themselves, or that the namespace already holds, raises ``DuplicateKeyError``, and a
        vector of another length than its model's stored vectors ``EmbeddingError``.
        """

    def modify(
        self, namespace: str, by: Lookup, value: str, build: Build, *, vector: Vector | None = None
    ) -> MemoryRecord:
        """Find a record, let ``build`` decide what to write, write it: one atomic step.

        ``build`` gets the current record, or ``None``, and must not change it. Its result
        stays in ``namespace``. If there is a current record, the result keeps its id and
        key and replaces it in place; if not, the result is added at the end and, when
        looked up by key, has that key. If ``build`` raises, nothing is written.
        The record's vector becomes what ``vector_after`` says. Returns the result.
        """

    def delete(self, namespace: str, id: str) -> bool:
        """Remove the record and its vector. ``False`` if there was none."""

    def purge_expired(self, now: datetime, *, limit: int) -> int:
        """Remove at most ``limit`` records, across namespaces, that have expired at ``now``, as
        ``pybodn._filter.is_expired`` tells, with their vectors, all at once. Returns how many.
        """

    def list(
        self, namespace: str, *, where: Filter, limit: int | None, offset: int
    ) -> builtins.list[MemoryRecord]:
        """The records ``where`` keeps, in the order added: skip ``offset``, return ``limit``.

        ``where`` keeps exactly the records for which ``where.matches(record)`` is true; how
        the store finds them is its own business. ``limit=None`` returns all the rest.
        """

    def count(self, namespace: str, *, where: Filter) -> int:
        """The number of records in ``namespace`` that ``where`` keeps, as ``list`` does."""

    def search_words(
        self, namespace: str, terms: Sequence[Term], *, where: Filter, limit: int
    ) -> builtins.list[SearchHit]:
        """The best ``limit`` records of ``namespace`` that ``where`` keeps and a term finds.

        A record scores as ``pybodn._search.bm25_scores`` scores it among every record of the
        store, other namespaces included; the best come first, and equal scores keep the order
        the records were added in. ``terms`` is never empty.
        """

    def search_vectors(
        self, namespace: str, query: Vector, *, where: Filter, limit: int
    ) -> builtins.list[SearchHit]:
        """The ``limit`` records of ``namespace`` that ``where`` keeps whose vectors of
        ``query.model`` are closest to ``query``.

        ``pybodn._vectors.best`` picks and scores them from those records, in the order added.
        No vectors of the model at all find nothing; a ``query`` of another length than them
        raises ``EmbeddingError``.
        """

    def last_seq(self) -> int:
        """The place of the last record in the order added, across namespaces; 0 if none."""

    def missing_vectors(
        self, model: str, *, after: int, until: int, limit: int
    ) -> builtins.list[MissingVector]:
        """At most ``limit`` records without a vector of ``model``, in the order added, from
        the place after ``after`` up to ``until``."""

    def write_vectors(self, vectors: Sequence[tuple[MissingVector, Vector]]) -> int:
        """Give each record its vector if it still holds the text the vector was made from,
        all at once, replacing any vector of another model. Returns how many it gave.

        A vector of another length than its model's stored vectors raises ``EmbeddingError``.
        """

    def close(self) -> None:
        """Release the storage. Safe to call again."""


def vector_after(
    current: MemoryRecord | None,
    record: MemoryRecord,
    vector: Vector | None,
    stored_size: Callable[[str], int | None],
) -> Vector | Unchanged | None:
    """The vector ``record`` has once written over ``current``, given ``vector``, made for it if
    any: that vector, ``UNCHANGED`` for the one it has, or ``None`` for none.

    A record only ever has a vector of the text it holds: the new vector if it was made from the
    text written, else the old one if the text stays, else none. A new vector must be as long as
    its model's stored vectors, which ``stored_size(model)`` tells, or ``EmbeddingError``.
    """
    if vector is not None and vector.text == record.content:
        check_size(vector, stored_size(vector.model))
        return vector
    if current is not None and current.content == record.content:
        return UNCHANGED
    return None


def check_size(vector: Vector, stored: int | None) -> None:
    """Raise ``EmbeddingError`` unless ``vector`` is ``stored`` bytes long, as its model's stored
    vectors are; ``None`` if there are none yet."""
    if stored is not None and stored != len(vector.data):
        raise EmbeddingError(
            f"model {vector.model!r} gave a vector of length {len(vector.data) // FLOAT_BYTES},"
            f" but its stored vectors have length {stored // FLOAT_BYTES}: give a model of"
            " another size another name"
        )


def check_sizes(vectors: Iterable[Vector], stored_size: Callable[[str], int | None]) -> None:
    """Raise ``EmbeddingError`` unless each vector is as long as its model's stored vectors, or,
    for a model with none stored yet, as its first vector here.

    ``stored_size(model)`` tells how many bytes long the stored vectors of ``model`` are.
    """
    sizes: dict[str, int | None] = {}
    for vector in vectors:
        if vector.model not in sizes:
            sizes[vector.model] = stored_size(vector.model)
        check_size(vector, sizes[vector.model])
        sizes[vector.model] = len(vector.data)


def text_and_key(record: MemoryRecord) -> TextAndKey:
    """What ``record`` shares with its copies."""
    return record.content, record.key


def split_batch(
    records: Sequence[MemoryRecord], copies: Mapping[TextAndKey, MemoryRecord] | None
) -> tuple[builtins.list[MemoryRecord], builtins.list[MemoryRecord]]:
    """The record a batch returns for each of ``records``, and the records it writes.

    Without ``copies``, it writes and returns every record. With them, a record that has a copy
    is not written, and its copy is returned in its place: the one ``copies`` holds for its text
    and key, else the first record of the batch with them. Raises ``DuplicateKeyError`` if two
    records it writes share a key.
    """
    if copies is None:
        returned = [*records]
    else:
        firsts = dict(copies)
        returned = [firsts.setdefault(text_and_key(record), record) for record in records]
    written = [record for record, kept in zip(records, returned, strict=True) if kept is record]
    _reject_repeated_keys(written)
    return returned, written


def _reject_repeated_keys(records: Iterable[MemoryRecord]) -> None:
    """Raise ``DuplicateKeyError`` if two of ``records`` share a key."""
    seen: set[str] = set()
    for record in records:
        if record.key in seen:
            raise DuplicateKeyError(f"key {record.key!r} is repeated within the batch")
        if record.key is not None:
            seen.add(record.key)


def reject_stored_keys(
    records: Iterable[MemoryRecord], is_stored: Callable[[str, str], bool]
) -> None:
    """Raise ``DuplicateKeyError`` for the first record whose key its namespace already has.

    ``is_stored(namespace, key)`` tells whether the store holds that key.
    """
    for record in records:
        if record.key is not None and is_stored(record.namespace, record.key):
            raise DuplicateKeyError(
                f"key {record.key!r} is already used in namespace {record.namespace!r};"
                " an archived or expired record may hold it: put() replaces it"
            )


def ensure_open(store: Store) -> None:
    """Raise ``EngineClosedError`` if ``store`` is closed."""
    if store.closed:
        raise EngineClosedError("the engine is closed")
