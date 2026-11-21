"""The records of one namespace, and what each operation does with them."""

import builtins
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import replace
from datetime import datetime
from functools import partial
from typing import Any, TypedDict, Unpack
from uuid import uuid4

from pybodn import _validation as validate
from pybodn._filter import Filter, Filters, Visibility, checked
from pybodn._mixing import mix
from pybodn._models import JSONValue, MemoryRecord, NewMemory, SearchHit, SearchMode
from pybodn._search import Term, query_terms
from pybodn._storage.base import NO_VECTORS, Lookup, Store, Vector, ensure_open, split_batch
from pybodn._words import words
from pybodn.errors import (
    ConcurrentUpdateError,
    ConfigurationError,
    RecordNotFoundError,
    ValidationError,
)

type Clock = Callable[[], datetime]

# Turns texts into the vectors an embedding model gives them, one for each, in the same order.
type Vectorize = Callable[[Sequence[str]], list[Vector]]

# How many of its best records each search gives a mixed search at the least: a record far down
# one search still rises if the other ranks it high.
_FROM_EACH = 50


class MemoryFields(TypedDict, total=False):
    """The optional fields of a memory, passed by name. Defaults live in ``NewMemory``."""

    metadata: Mapping[str, JSONValue]
    tags: Sequence[str]
    importance: float
    source: str | None
    expires_at: datetime | None


class NewMemoryFields(MemoryFields, total=False):
    """The optional fields ``add`` takes."""

    key: str | None


class MemoryChanges(MemoryFields, total=False):
    """The fields ``update`` can change."""

    content: str


# Each field a caller sets, and the check that turns the value given into the value stored.
# Adding and changing a record both use this one table.
_FIELD_CHECKS: dict[str, Callable[[object], Any]] = {
    "content": partial(validate.text, "content"),
    "key": validate.key,
    "metadata": validate.metadata,
    "tags": validate.tags,
    "importance": validate.importance,
    "source": validate.source,
    "expires_at": validate.expiry,
}


class Collection:
    """The records of one namespace. Get it from ``MemoryEngine.collection()``.

    Archived records, and those whose ``expires_at`` has come, stay in the store but are hidden:
    every read leaves them out unless ``include_archived=True`` or ``include_expired=True`` asks
    for them. Writes still reach them, and a hidden record keeps its key until it is deleted.

    With an embedding model, every write of a new text asks the model for its vector first,
    outside the store's lock, and writes nothing if the model fails: an error the model raises
    passes through as it is, and an answer it gives that cannot be used raises
    ``EmbeddingError``. The other errors every method may raise are those ``MemoryEngine`` names.
    """

    __slots__ = ("_clock", "_namespace", "_store", "_vectorize")

    def __init__(
        self, store: Store, namespace: str, clock: Clock, vectorize: Vectorize | None
    ) -> None:
        """Internal: use ``MemoryEngine.collection()``. ``vectorize`` is ``None`` without a
        model."""
        self._store = store
        self._namespace = namespace
        self._clock = clock
        self._vectorize = vectorize

    @property
    def namespace(self) -> str:
        """The namespace of every record in this collection."""
        return self._namespace

    def add(
        self, content: str, *, dedupe: bool = False, **fields: Unpack[NewMemoryFields]
    ) -> MemoryRecord:
        """Add a record of ``content`` with the fields passed by name, those of ``NewMemory``.

        The same as ``add_many([NewMemory(content, **fields)], dedupe=dedupe)[0]``, with the same
        errors.
        """
        ensure_open(self._store)
        validate.reject_unknown("add", fields, NewMemoryFields.__optional_keys__)
        return self.add_many([NewMemory(content, **fields)], dedupe=dedupe)[0]

    def add_many(
        self, memories: Iterable[NewMemory], *, dedupe: bool = False
    ) -> builtins.list[MemoryRecord]:
        """Add every memory of ``memories``, any iterable of ``NewMemory``, or none if any of
        them fails. Returns a record for each, in the same order.

        With ``dedupe=True``, a memory is not added if the namespace shows a record with the
        same text and the same key, or both without one, or if an earlier memory of the batch
        has them. That record comes back in its place, as stored: the memory's other fields are
        not applied. Texts must match in every character. Archived and expired records are no
        copies, but still hold their keys.

        Raises ``DuplicateKeyError`` if a key is already used in the namespace, even by a hidden
        record, or two of the memories written share one.
        """
        ensure_open(self._store)
        checked_dedupe = validate.flag("dedupe", dedupe)
        makers = [_prepare(self._namespace, memory) for memory in validate.memories(memories)]
        now = self._clock()
        records = [
            make(id=uuid4().hex, created_at=now, updated_at=now, version=1) for make in makers
        ]
        visible = checked("add_many", {}, now=now) if checked_dedupe else None  # where copies are
        vectors = self._vectors_of(records, visible)
        return self._store.insert(self._namespace, records, vectors=vectors, dedupe=visible)

    def put(self, key: str, content: str, **fields: Unpack[MemoryFields]) -> MemoryRecord:
        """Create the record with ``key``, or replace every field of it, from ``content`` and
        the fields passed by name: those of ``NewMemory`` other than ``key``.

        A field not passed gets its default. A replaced record keeps its id, creation time and
        place in ``list()``, and comes back from the archive. Its old values stay in the file
        until ``MemoryEngine.erase_deleted()``.
        """
        ensure_open(self._store)
        validate.reject_unknown("put", fields, MemoryFields.__optional_keys__)
        checked_key = validate.text("key", key)
        make = _prepare(self._namespace, NewMemory(content, key=checked_key, **fields))
        text = make.keywords["content"]
        vector = self._vector_for("key", checked_key, text, creates=True)

        def build(current: MemoryRecord | None) -> MemoryRecord:
            now = self._clock()
            if current is None:
                return make(id=uuid4().hex, created_at=now, updated_at=now, version=1)
            return make(
                id=current.id,
                created_at=current.created_at,
                updated_at=now,
                version=current.version + 1,
            )

        return self._store.modify(self._namespace, "key", checked_key, build, vector=vector)

    def get(self, id: str, **options: Unpack[Visibility]) -> MemoryRecord | None:
        """The record with ``id``, or ``None``. A hidden one comes only if
        ``include_archived=True`` or ``include_expired=True`` asks for it."""
        ensure_open(self._store)
        return self._get("get", "id", validate.text("id", id), options)

    def get_by_key(self, key: str, **options: Unpack[Visibility]) -> MemoryRecord | None:
        """The record with ``key``, or ``None``. A hidden one comes only if
        ``include_archived=True`` or ``include_expired=True`` asks for it."""
        ensure_open(self._store)
        return self._get("get_by_key", "key", validate.text("key", key), options)

    def update(
        self, id: str, *, expected_version: int | None = None, **changes: Unpack[MemoryChanges]
    ) -> MemoryRecord:
        """Change only the fields passed by name, at least one: any of ``NewMemory`` other than
        ``key``. ``source=None`` clears the source, and ``expires_at=None`` the expiry. An
        archived record stays archived. The old values stay in the file until
        ``MemoryEngine.erase_deleted()``.

        With ``expected_version``, the record changes only if it is still at that version.
        Raises ``RecordNotFoundError`` if there is no such record, and ``ConcurrentUpdateError``
        if ``expected_version`` is not the record's version.
        """
        ensure_open(self._store)
        checked = _check_changes(changes)
        checked_id = validate.text("id", id)
        expected = validate.expected_version(expected_version)
        text = checked.get("content")
        vector = None if text is None else self._vector_for("id", checked_id, text, creates=False)

        def build(current: MemoryRecord | None) -> MemoryRecord:
            current = self._found(current, checked_id)
            if expected is not None and current.version != expected:
                raise ConcurrentUpdateError(
                    f"record {checked_id!r} is at version {current.version}, not {expected}"
                )
            return replace(
                current, **checked, updated_at=self._clock(), version=current.version + 1
            )

        return self._store.modify(self._namespace, "id", checked_id, build, vector=vector)

    def archive(self, id: str) -> MemoryRecord:
        """Hide the record from every read until ``restore()`` or ``put()``. It stays stored,
        and ``MemoryEngine.erase_deleted()`` keeps it. Archiving it again changes nothing.
        Raises ``RecordNotFoundError`` if there is no such record."""
        ensure_open(self._store)
        return self._set_archived(validate.text("id", id), archived=True)

    def restore(self, id: str) -> MemoryRecord:
        """Bring the record back from the archive. Restoring one not archived changes nothing.
        Raises ``RecordNotFoundError`` if there is no such record."""
        ensure_open(self._store)
        return self._set_archived(validate.text("id", id), archived=False)

    def delete(self, id: str) -> bool:
        """Remove the record and return ``True``, or ``False`` if there was none, so repeating
        is safe. Its data stays in the file until ``MemoryEngine.erase_deleted()``."""
        ensure_open(self._store)
        return self._store.delete(self._namespace, validate.text("id", id))

    def list(
        self, *, limit: int | None = None, offset: int = 0, **filters: Unpack[Filters]
    ) -> builtins.list[MemoryRecord]:
        """Records in the order they were added. ``limit``, ``None`` or an integer from 1, and
        ``offset``, an integer from 0, select a page.

        Filters keep only the records that pass every one given; ``None`` sets no condition.

        - ``tags``: has all these tags (at most 100).
        - ``metadata``: each of these top-level fields (at most 100) is there and equals its
          value as JSON: ``True`` is not ``1``, and ``None`` means set to ``None``.
        - ``min_importance``: importance at least this.
        - ``created_since``, ``created_before``: created at or after, and before, these
          moments; each a ``datetime`` with a time zone.

        ``include_archived=True`` and ``include_expired=True`` keep those hidden records too.
        """
        ensure_open(self._store)
        return self._store.list(
            self._namespace,
            where=self._visible("list", filters),
            limit=validate.limit(limit),
            offset=validate.offset(offset),
        )

    def count(self, **filters: Unpack[Filters]) -> int:
        """The number of records, or of those that pass the filters ``list`` takes."""
        ensure_open(self._store)
        return self._store.count(self._namespace, where=self._visible("count", filters))

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        mode: SearchMode = "auto",
        **filters: Unpack[Filters],
    ) -> builtins.list[SearchHit]:
        """The records that best match ``query``, best first: at most ``limit`` of them, an
        integer from 1.

        ``mode="text"`` searches by words. A record matches if it holds any word of the query,
        in any case, with or without stress marks and apostrophes. A word also finds its forms
        with another ending: "алергії" finds "алергія". More of the query's words, rarer words
        and the exact forms rank a record higher. A query of more than 100 different words
        raises ``ValidationError``.

        ``mode="vector"`` searches by meaning, with the engine's embedding model, and raises
        ``ConfigurationError`` without one. The records whose vectors are closest to the query's
        come first, however far, each scored by its similarity to the query from -1 to 1. A
        record without a vector of the model is not found: see ``MemoryEngine.embed_missing()``.

        ``mode="auto"``, the default, searches by words, and with an embedding model by meaning
        too, and mixes the two by reciprocal rank fusion: a record found both ways rises, and
        scores are small, such as 0.03. If the model fails, the search raises its error.

        Equal scores keep the order the records were added in; in a mixed search, the order of
        the search by words. A query without words finds nothing. Filters are those ``list``
        takes.
        """
        ensure_open(self._store)
        where = self._visible("search", filters)
        text = validate.query(query)
        checked_limit = validate.search_limit(limit)
        checked_mode = validate.mode(mode)
        if checked_mode == "vector":
            return self._search_by_meaning(text, where, checked_limit)
        terms = query_terms(text)
        if not terms:
            return []
        if checked_mode == "auto" and self._vectorize is not None:
            [vector] = self._vectorize([text])  # first, so that the model never holds up the store
            return self._search_both_ways(terms, vector, where, checked_limit)
        return self._store.search_words(self._namespace, terms, where=where, limit=checked_limit)

    def _visible(
        self,
        method: str,
        filters: Mapping[str, object],
        known: frozenset[str] = Filters.__optional_keys__,
    ) -> Filter:
        """The filters passed to ``method``, which also leave out the records hidden now."""
        return checked(method, filters, now=self._clock(), known=known)

    def _get(
        self, method: str, by: Lookup, value: str, options: Mapping[str, object]
    ) -> MemoryRecord | None:
        """The record ``by`` finds, if the options passed to ``method`` let it be seen."""
        where = self._visible(method, options, Visibility.__optional_keys__)
        record = self._store.get(self._namespace, by, value)
        return record if record is not None and where.matches(record) else None

    def _set_archived(self, id: str, *, archived: bool) -> MemoryRecord:
        """Archive or restore the record; one already so is returned as it is."""

        def build(current: MemoryRecord | None) -> MemoryRecord:
            current = self._found(current, id)
            if (current.archived_at is not None) == archived:
                return current
            now = self._clock()
            return replace(
                current,
                archived_at=now if archived else None,
                updated_at=now,
                version=current.version + 1,
            )

        return self._store.modify(self._namespace, "id", id, build)

    def _found(self, current: MemoryRecord | None, id: str) -> MemoryRecord:
        """``current``, the record a write found by ``id``; ``RecordNotFoundError`` if none."""
        if current is None:
            raise RecordNotFoundError(f"no record with id {id!r} in namespace {self._namespace!r}")
        return current

    def _search_by_meaning(self, query: str, where: Filter, limit: int) -> builtins.list[SearchHit]:
        if self._vectorize is None:
            raise ConfigurationError(
                "mode='vector' searches by meaning and needs an embedding model; this engine"
                " has none"
            )
        if not words(query):
            return []
        [vector] = self._vectorize([query])
        return self._store.search_vectors(self._namespace, vector, where=where, limit=limit)

    def _search_both_ways(
        self, terms: Sequence[Term], vector: Vector, where: Filter, limit: int
    ) -> builtins.list[SearchHit]:
        """Search by words and by meaning, each on its own, and mix what they find."""
        each = max(_FROM_EACH, limit)
        by_words = self._store.search_words(self._namespace, terms, where=where, limit=each)
        by_meaning = self._store.search_vectors(self._namespace, vector, where=where, limit=each)
        return mix(by_words, by_meaning, limit=limit)

    def _vectors_of(
        self, records: Sequence[MemoryRecord], dedupe: Filter | None
    ) -> Mapping[str, Vector]:
        """The vector of each record a batch will write, by the record's id; none without a model.

        With ``dedupe``, a record whose copy is among the records it keeps gets none. Copies are
        looked for here, before the model, so that it is asked only about new texts; the store
        looks again as it writes, since another writer may add or hide one meanwhile.
        """
        if self._vectorize is None:
            return NO_VECTORS
        copies = (
            None
            if dedupe is None
            else self._store.find_copies(self._namespace, records, where=dedupe)
        )
        _, written = split_batch(records, copies)
        vectors = self._vectorize([record.content for record in written])
        return {record.id: vector for record, vector in zip(written, vectors, strict=True)}

    def _vector_for(self, by: Lookup, value: str, text: str, *, creates: bool) -> Vector | None:
        """The vector of ``text`` for a write over the record ``by`` finds, if it needs one: there
        is a model, and that record holds another text, or there is none and the write
        ``creates`` one.

        The record is read before the write, so that the model never holds up the store. If it
        changes in between, the store's ``vector_after`` still keeps a vector only with its text.
        """
        if self._vectorize is None:
            return None
        current = self._store.get(self._namespace, by, value)
        if current is None and not creates:  # the write will fail: there is nothing to change
            return None
        if current is not None and current.content == text:  # the record keeps its vector
            return None
        [vector] = self._vectorize([text])
        return vector


def _prepare(namespace: str, memory: NewMemory) -> partial[MemoryRecord]:
    """Check ``memory``. The result makes its record, never archived, from an id, two times and
    a version."""
    checked = {name: check(getattr(memory, name)) for name, check in _FIELD_CHECKS.items()}
    return partial(MemoryRecord, namespace=namespace, archived_at=None, **checked)


def _check_changes(changes: Mapping[str, object]) -> dict[str, Any]:
    validate.reject_unknown("update", changes, MemoryChanges.__optional_keys__)
    if not changes:
        raise ValidationError("update needs at least one field to change")
    return {name: check(changes[name]) for name, check in _FIELD_CHECKS.items() if name in changes}
