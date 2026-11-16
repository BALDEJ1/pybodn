"""A store that keeps records in this process's memory."""

import builtins
import copy
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime

from pybodn._filter import EVERYTHING, Filter, is_expired
from pybodn._models import MemoryRecord, SearchHit
from pybodn._search import Term, bm25_scores
from pybodn._storage.base import (
    NO_VECTORS,
    Build,
    Lookup,
    MissingVector,
    Unchanged,
    Vector,
    check_size,
    check_sizes,
    ensure_open,
    reject_stored_keys,
    vector_after,
)
from pybodn._words import words


@dataclass(slots=True)
class _Namespace:
    """Records by id, in the order added; the id of each key; the place of each record in the
    order added across namespaces; and the vector of each record that has one."""

    records: dict[str, MemoryRecord] = field(default_factory=dict)
    ids_by_key: dict[str, str] = field(default_factory=dict)
    seqs: dict[str, int] = field(default_factory=dict)
    vectors: dict[str, Vector] = field(default_factory=dict)


class InMemoryStore:
    """Records in a dict per namespace, behind one lock, so threads can share it.

    Records are copied on the way in and out: changing the ``metadata`` of a record
    it returned, or of one passed to it, never changes what is stored.
    """

    __slots__ = ("_closed", "_last_seq", "_lock", "_namespaces")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._namespaces: dict[str, _Namespace] = {}
        self._last_seq = 0  # the place of the last record ever added
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def get(self, namespace: str, by: Lookup, value: str) -> MemoryRecord | None:
        with self._lock:
            ensure_open(self)
            record = self._find(namespace, by, value)
            return None if record is None else _copy(record)

    def insert(
        self, records: Sequence[MemoryRecord], *, vectors: Mapping[str, Vector] = NO_VECTORS
    ) -> None:
        copies = [_copy(record) for record in records]  # a copy that fails writes nothing
        with self._lock:
            ensure_open(self)
            reject_stored_keys(copies, self._is_key_stored)
            check_sizes(vectors.values(), self._vector_size)
            for record in copies:
                self._write(record, vectors.get(record.id))

    def modify(
        self, namespace: str, by: Lookup, value: str, build: Build, *, vector: Vector | None = None
    ) -> MemoryRecord:
        with self._lock:
            ensure_open(self)
            current = self._find(namespace, by, value)
            record = build(current)
            after = vector_after(current, record, vector, self._vector_size)
            self._write(_copy(record), after)
            return record

    def delete(self, namespace: str, id: str) -> bool:
        with self._lock:
            ensure_open(self)
            return self._remove(namespace, id)

    def purge_expired(self, now: datetime, *, limit: int) -> int:
        with self._lock:
            ensure_open(self)
            # Looks through every record for each batch: this store is for tests and small data.
            expired = [
                (namespace, record.id)
                for namespace, space in self._namespaces.items()
                for record in space.records.values()
                if is_expired(record, now)
            ][:limit]
            for namespace, record_id in expired:
                self._remove(namespace, record_id)
            return len(expired)

    def list(
        self, namespace: str, *, where: Filter, limit: int | None, offset: int
    ) -> builtins.list[MemoryRecord]:
        with self._lock:
            ensure_open(self)
            stop = None if limit is None else offset + limit
            return [_copy(record) for record in self._kept(namespace, where)[offset:stop]]

    def count(self, namespace: str, *, where: Filter) -> int:
        with self._lock:
            ensure_open(self)
            return len(self._kept(namespace, where))

    def search_words(
        self, namespace: str, terms: Sequence[Term], *, where: Filter, limit: int
    ) -> builtins.list[SearchHit]:
        with self._lock:
            ensure_open(self)
            # Words are prepared afresh for every search: this store is for tests and small data.
            records = [
                record for space in self._namespaces.values() for record in space.records.values()
            ]
            scores = bm25_scores(terms, [words(record.content) for record in records])
            found = [
                (score, record)
                for record, score in zip(records, scores, strict=True)
                if score is not None and record.namespace == namespace and where.matches(record)
            ]
            found.sort(key=lambda hit: hit[0], reverse=True)  # stable: ties keep the order added
            return [SearchHit(record=_copy(record), score=score) for score, record in found[:limit]]

    def search_vectors(
        self, namespace: str, query: Vector, *, where: Filter, limit: int
    ) -> builtins.list[SearchHit]:
        with self._lock:
            ensure_open(self)
            from pybodn import _vectors  # noqa: PLC0415 - NumPy, only once there is a model

            check_size(query, self._vector_size(query.model))
            space = self._namespaces.get(namespace, _Namespace())
            candidates = (
                (record, vector.data)
                for record in self._kept(namespace, where)
                if (vector := space.vectors.get(record.id)) is not None
                and vector.model == query.model
            )
            found = _vectors.best(query, candidates, limit=limit)
            return [SearchHit(record=_copy(record), score=score) for score, record in found]

    def last_seq(self) -> int:
        with self._lock:
            ensure_open(self)
            # A namespace's last record has its highest place.
            return max(
                (next(reversed(space.seqs.values())) for space in self._namespaces.values()),
                default=0,
            )

    def missing_vectors(
        self, model: str, *, after: int, until: int, limit: int
    ) -> builtins.list[MissingVector]:
        with self._lock:
            ensure_open(self)
            missing = [
                MissingVector(seq, namespace, record_id, space.records[record_id].content)
                for namespace, space in self._namespaces.items()
                for record_id, seq in space.seqs.items()
                if after < seq <= until
                and ((vector := space.vectors.get(record_id)) is None or vector.model != model)
            ]
            missing.sort(key=lambda record: record.seq)
            return missing[:limit]

    def write_vectors(self, vectors: Sequence[tuple[MissingVector, Vector]]) -> int:
        with self._lock:
            ensure_open(self)
            check_sizes((vector for _, vector in vectors), self._vector_size)
            given = 0
            for missing, vector in vectors:
                space = self._namespaces.get(missing.namespace, _Namespace())
                record = space.records.get(missing.id)
                if record is not None and record.content == vector.text:
                    space.vectors[missing.id] = vector
                    given += 1
            return given

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._namespaces.clear()

    def _kept(self, namespace: str, where: Filter) -> builtins.list[MemoryRecord]:
        """The records of ``namespace`` that ``where`` keeps, in the order added, uncopied."""
        space = self._namespaces.get(namespace)
        if space is None:
            return []
        records = space.records.values()
        if where == EVERYTHING:  # no filter to run each record through
            return [*records]
        return [record for record in records if where.matches(record)]

    def _remove(self, namespace: str, id: str) -> bool:
        """Remove the record and its vector. ``False`` if there was none."""
        space = self._namespaces.get(namespace)
        if space is None or id not in space.records:
            return False
        record = space.records.pop(id)
        if record.key is not None:
            del space.ids_by_key[record.key]
        del space.seqs[id]
        space.vectors.pop(id, None)
        if not space.records:
            del self._namespaces[namespace]
        return True

    def _find(self, namespace: str, by: Lookup, value: str) -> MemoryRecord | None:
        space = self._namespaces.get(namespace)
        if space is None:
            return None
        record_id = value if by == "id" else space.ids_by_key.get(value)
        return None if record_id is None else space.records.get(record_id)

    def _is_key_stored(self, namespace: str, key: str) -> bool:
        return self._find(namespace, "key", key) is not None

    def _vector_size(self, model: str) -> int | None:
        """How many bytes long the stored vectors of ``model`` are; ``None`` if there are none.

        Looks through every vector: this store is for tests and small data.
        """
        sizes = (
            len(vector.data)
            for space in self._namespaces.values()
            for vector in space.vectors.values()
            if vector.model == model
        )
        return next(sizes, None)

    def _write(self, record: MemoryRecord, vector: Vector | Unchanged | None) -> None:
        """Add ``record`` at the end, or replace the record with its id in place, and give it
        ``vector``: a new one, the one it has, or none."""
        space = self._namespaces.setdefault(record.namespace, _Namespace())
        if record.id not in space.records:
            self._last_seq += 1
            space.seqs[record.id] = self._last_seq
        space.records[record.id] = record
        if record.key is not None:
            space.ids_by_key[record.key] = record.id
        if isinstance(vector, Vector):
            space.vectors[record.id] = vector
        elif vector is None:
            space.vectors.pop(record.id, None)


def _copy(record: MemoryRecord) -> MemoryRecord:
    """A copy that shares nothing changeable with ``record``."""
    return replace(record, metadata=copy.deepcopy(record.metadata))
