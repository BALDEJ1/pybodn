"""The records of one namespace, and what each operation does with them."""

import builtins
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import replace
from datetime import datetime
from functools import partial
from typing import Any, TypedDict, Unpack
from uuid import uuid4

from pybodn import _validation as validate
from pybodn._filter import Filter, Filters, checked
from pybodn._models import JSONValue, MemoryRecord, NewMemory
from pybodn._storage.base import Store, ensure_open
from pybodn.errors import (
    ConcurrentUpdateError,
    DuplicateKeyError,
    RecordNotFoundError,
    ValidationError,
)

type Clock = Callable[[], datetime]


class MemoryFields(TypedDict, total=False):
    """The optional fields of a memory, passed by name. Defaults live in ``NewMemory``."""

    metadata: Mapping[str, JSONValue]
    tags: Sequence[str]
    importance: float
    source: str | None


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
}


class Collection:
    """The records of one namespace. Get it from ``MemoryEngine.collection()``."""

    __slots__ = ("_clock", "_namespace", "_store")

    def __init__(self, store: Store, namespace: str, clock: Clock) -> None:
        """Internal: use ``MemoryEngine.collection()``."""
        self._store = store
        self._namespace = namespace
        self._clock = clock

    @property
    def namespace(self) -> str:
        """The namespace of every record in this collection."""
        return self._namespace

    def add(self, content: str, **fields: Unpack[NewMemoryFields]) -> MemoryRecord:
        """Add a record. The same as ``add_many([NewMemory(content, ...)])[0]``."""
        ensure_open(self._store)
        _reject_unknown("add", fields, NewMemoryFields.__optional_keys__)
        return self.add_many([NewMemory(content, **fields)])[0]

    def add_many(self, memories: Iterable[NewMemory]) -> builtins.list[MemoryRecord]:
        """Add every memory, or none if any of them fails. Returns them in the same order."""
        ensure_open(self._store)
        makers = [_prepare(self._namespace, memory) for memory in validate.memories(memories)]
        now = self._clock()
        records = [
            make(id=uuid4().hex, created_at=now, updated_at=now, version=1) for make in makers
        ]
        _reject_repeated_keys(records)
        self._store.insert(records)
        return records

    def put(self, key: str, content: str, **fields: Unpack[MemoryFields]) -> MemoryRecord:
        """Create the record with ``key``, or replace every field of it.

        A field not passed gets its default. A replaced record keeps its id, creation
        time and place in ``list()``.
        """
        ensure_open(self._store)
        _reject_unknown("put", fields, MemoryFields.__optional_keys__)
        checked_key = validate.text("key", key)
        make = _prepare(self._namespace, NewMemory(content, key=checked_key, **fields))

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

        return self._store.modify(self._namespace, "key", checked_key, build)

    def get(self, id: str) -> MemoryRecord | None:
        """The record with this id, or ``None``."""
        ensure_open(self._store)
        return self._store.get(self._namespace, "id", validate.text("id", id))

    def get_by_key(self, key: str) -> MemoryRecord | None:
        """The record with this key, or ``None``."""
        ensure_open(self._store)
        return self._store.get(self._namespace, "key", validate.text("key", key))

    def update(
        self, id: str, *, expected_version: int | None = None, **changes: Unpack[MemoryChanges]
    ) -> MemoryRecord:
        """Change only the fields passed. ``source=None`` clears the source.

        Raises ``RecordNotFoundError`` if there is no such record, and
        ``ConcurrentUpdateError`` if ``expected_version`` is not the record's version.
        """
        ensure_open(self._store)
        checked = _check_changes(changes)
        checked_id = validate.text("id", id)
        expected = validate.expected_version(expected_version)

        def build(current: MemoryRecord | None) -> MemoryRecord:
            if current is None:
                raise RecordNotFoundError(
                    f"no record with id {checked_id!r} in namespace {self._namespace!r}"
                )
            if expected is not None and current.version != expected:
                raise ConcurrentUpdateError(
                    f"record {checked_id!r} is at version {current.version}, not {expected}"
                )
            return replace(
                current, **checked, updated_at=self._clock(), version=current.version + 1
            )

        return self._store.modify(self._namespace, "id", checked_id, build)

    def delete(self, id: str) -> bool:
        """Remove the record. ``False`` if there was none, so repeating is safe."""
        ensure_open(self._store)
        return self._store.delete(self._namespace, validate.text("id", id))

    def list(
        self, *, limit: int | None = None, offset: int = 0, **filters: Unpack[Filters]
    ) -> builtins.list[MemoryRecord]:
        """Records in the order they were added. ``limit`` and ``offset`` select a page.

        Filters keep only the records that pass every one given:

        - ``tags``: has every one of these tags, at most 100;
        - ``metadata``: each of these top-level fields, at most 100, is there and equals its
          value, as JSON compares them: ``True`` is not ``1``, ``1`` is ``1.0``, and
          ``None`` is a field set to ``None``, not a missing one;
        - ``min_importance``: importance at least this;
        - ``created_since`` and ``created_before``: created at or after, and before, these
          moments, each a ``datetime`` with a time zone.
        """
        ensure_open(self._store)
        checked_limit, checked_offset = validate.limit(limit), validate.offset(offset)
        where = _check_filters("list", filters)
        return self._store.list(
            self._namespace, where=where, limit=checked_limit, offset=checked_offset
        )

    def count(self, **filters: Unpack[Filters]) -> int:
        """The number of records, or of those that pass the filters, as ``list`` takes them."""
        ensure_open(self._store)
        return self._store.count(self._namespace, where=_check_filters("count", filters))


def _prepare(namespace: str, memory: NewMemory) -> partial[MemoryRecord]:
    """Check ``memory``. The result makes its record from an id, two times and a version."""
    checked = {name: check(getattr(memory, name)) for name, check in _FIELD_CHECKS.items()}
    return partial(MemoryRecord, namespace=namespace, **checked)


def _reject_repeated_keys(records: Iterable[MemoryRecord]) -> None:
    """Raise ``DuplicateKeyError`` if two records of one batch share a key."""
    seen: set[str] = set()
    for record in records:
        if record.key in seen:
            raise DuplicateKeyError(f"key {record.key!r} is repeated within the batch")
        if record.key is not None:
            seen.add(record.key)


def _reject_unknown(method: str, fields: Mapping[str, object], known: frozenset[str]) -> None:
    """Raise the ``TypeError`` Python raises for a keyword a function does not take."""
    unknown = sorted(fields.keys() - known)
    if unknown:
        raise TypeError(f"{method}() got an unexpected keyword argument {unknown[0]!r}")


def _check_filters(method: str, filters: Mapping[str, object]) -> Filter:
    _reject_unknown(method, filters, Filters.__optional_keys__)
    return checked(filters)


def _check_changes(changes: Mapping[str, object]) -> dict[str, Any]:
    _reject_unknown("update", changes, MemoryChanges.__optional_keys__)
    if not changes:
        raise ValidationError("update needs at least one field to change")
    return {name: check(changes[name]) for name, check in _FIELD_CHECKS.items() if name in changes}
