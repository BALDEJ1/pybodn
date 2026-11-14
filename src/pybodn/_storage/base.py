"""The contract every storage implements.

A store makes no decisions: it only reads and writes, atomically. Checks, ids, times,
versions and what ``put`` and ``update`` mean live above it, once for every store.

Errors: whatever ``build`` raises, and any pybodn error, passes through unchanged. A
failure of the storage itself becomes ``StorageError`` with the original as its cause.
Any call after ``close()`` raises ``EngineClosedError``.
"""

import builtins
from collections.abc import Callable, Iterable, Sequence
from typing import Literal, Protocol

from pybodn._filter import Filter
from pybodn._models import MemoryRecord
from pybodn.errors import DuplicateKeyError, EngineClosedError

# The field that ``value`` is matched against.
type Lookup = Literal["id", "key"]

# Turns the current record, or ``None`` if there is none, into the record to write.
type Build = Callable[[MemoryRecord | None], MemoryRecord]


class Store(Protocol):
    """Atomic reads and writes of records, kept apart by namespace."""

    @property
    def closed(self) -> bool:
        """Whether ``close()`` has been called."""

    def get(self, namespace: str, by: Lookup, value: str) -> MemoryRecord | None:
        """The record of ``namespace`` whose ``by`` equals ``value``, or ``None``."""

    def insert(self, records: Sequence[MemoryRecord]) -> None:
        """Write every record, or none of them.

        A key already used in the namespace raises ``DuplicateKeyError``. The caller sees to
        it that ``records`` never repeat a key among themselves.
        """

    def modify(self, namespace: str, by: Lookup, value: str, build: Build) -> MemoryRecord:
        """Find a record, let ``build`` decide what to write, write it: one atomic step.

        ``build`` gets the current record, or ``None``, and must not change it. Its result
        stays in ``namespace``. If there is a current record, the result keeps its id and
        key and replaces it in place; if not, the result is added at the end and, when
        looked up by key, has that key. If ``build`` raises, nothing is written.
        Returns the result.
        """

    def delete(self, namespace: str, id: str) -> bool:
        """Remove the record. ``False`` if there was none."""

    def list(
        self, namespace: str, *, where: Filter, limit: int | None, offset: int
    ) -> builtins.list[MemoryRecord]:
        """The records ``where`` keeps, in the order they were added: skip ``offset``,
        return up to ``limit``.

        ``limit=None`` returns all the rest.
        """

    def count(self, namespace: str, *, where: Filter) -> int:
        """The number of records in ``namespace`` that ``where`` keeps."""

    def close(self) -> None:
        """Release the storage. Safe to call again."""


def reject_stored_keys(
    records: Iterable[MemoryRecord], is_stored: Callable[[str, str], bool]
) -> None:
    """Raise ``DuplicateKeyError`` for the first record whose key its namespace already has.

    ``is_stored(namespace, key)`` tells whether the store holds that key.
    """
    for record in records:
        if record.key is not None and is_stored(record.namespace, record.key):
            raise DuplicateKeyError(
                f"key {record.key!r} is already used in namespace {record.namespace!r}"
            )


def ensure_open(store: Store) -> None:
    """Raise ``EngineClosedError`` if ``store`` is closed."""
    if store.closed:
        raise EngineClosedError("the engine is closed")
