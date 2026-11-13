"""A store that keeps records in this process's memory."""

import builtins
import copy
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

from pybodn._models import MemoryRecord
from pybodn._storage.base import Build, Lookup, ensure_open, reject_stored_keys


@dataclass(slots=True)
class _Namespace:
    """Records by id, in the order added; and the id of each key."""

    records: dict[str, MemoryRecord] = field(default_factory=dict)
    ids_by_key: dict[str, str] = field(default_factory=dict)


class InMemoryStore:
    """Records in a dict per namespace, behind one lock, so threads can share it.

    Records are copied on the way in and out: changing the ``metadata`` of a record
    it returned, or of one passed to it, never changes what is stored.
    """

    __slots__ = ("_closed", "_lock", "_namespaces")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._namespaces: dict[str, _Namespace] = {}
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def get(self, namespace: str, by: Lookup, value: str) -> MemoryRecord | None:
        with self._lock:
            ensure_open(self)
            record = self._find(namespace, by, value)
            return None if record is None else _copy(record)

    def insert(self, records: Sequence[MemoryRecord]) -> None:
        copies = [_copy(record) for record in records]  # a copy that fails writes nothing
        with self._lock:
            ensure_open(self)
            reject_stored_keys(copies, self._is_key_stored)
            for record in copies:
                self._write(record)

    def modify(self, namespace: str, by: Lookup, value: str, build: Build) -> MemoryRecord:
        with self._lock:
            ensure_open(self)
            record = build(self._find(namespace, by, value))
            self._write(_copy(record))
            return record

    def delete(self, namespace: str, id: str) -> bool:
        with self._lock:
            ensure_open(self)
            space = self._namespaces.get(namespace)
            if space is None or id not in space.records:
                return False
            record = space.records.pop(id)
            if record.key is not None:
                del space.ids_by_key[record.key]
            if not space.records:
                del self._namespaces[namespace]
            return True

    def list(
        self, namespace: str, *, limit: int | None, offset: int
    ) -> builtins.list[MemoryRecord]:
        with self._lock:
            ensure_open(self)
            space = self._namespaces.get(namespace)
            if space is None:
                return []
            stop = None if limit is None else offset + limit
            return [_copy(record) for record in [*space.records.values()][offset:stop]]

    def count(self, namespace: str) -> int:
        with self._lock:
            ensure_open(self)
            space = self._namespaces.get(namespace)
            return 0 if space is None else len(space.records)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._namespaces.clear()

    def _find(self, namespace: str, by: Lookup, value: str) -> MemoryRecord | None:
        space = self._namespaces.get(namespace)
        if space is None:
            return None
        record_id = value if by == "id" else space.ids_by_key.get(value)
        return None if record_id is None else space.records.get(record_id)

    def _is_key_stored(self, namespace: str, key: str) -> bool:
        return self._find(namespace, "key", key) is not None

    def _write(self, record: MemoryRecord) -> None:
        """Add ``record`` at the end, or replace the record with its id in place."""
        space = self._namespaces.setdefault(record.namespace, _Namespace())
        space.records[record.id] = record
        if record.key is not None:
            space.ids_by_key[record.key] = record.id


def _copy(record: MemoryRecord) -> MemoryRecord:
    """A copy that shares nothing changeable with ``record``."""
    return replace(record, metadata=copy.deepcopy(record.metadata))
