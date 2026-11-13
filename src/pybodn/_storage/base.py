"""The contract every storage implements.

A store makes no decisions: it only reads and writes, atomically. Checks, ids, times,
versions and what ``put`` and ``update`` mean live above it, once for every store.

Errors: pybodn errors, including those raised by ``build``, pass through unchanged. A
failure of the storage itself becomes ``StorageError`` with the original as its cause.
Any call after ``close()`` raises ``EngineClosedError``.
"""

import builtins
from collections.abc import Callable, Sequence
from typing import Literal, Protocol

from pybodn._models import MemoryRecord

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

        A key already used in the namespace, or repeated among ``records``, raises
        ``DuplicateKeyError``.
        """

    def modify(self, namespace: str, by: Lookup, value: str, build: Build) -> MemoryRecord:
        """Find a record, let ``build`` decide what to write, write it: one atomic step.

        ``build`` gets the current record, or ``None``. It must not change that record,
        and its result keeps the record's id, namespace and key. The result replaces the
        current record in place or, if there is none, is added at the end. If ``build``
        raises, nothing is written. Returns the result.
        """

    def delete(self, namespace: str, id: str) -> bool:
        """Remove the record. ``False`` if there was none."""

    def list(
        self, namespace: str, *, limit: int | None, offset: int
    ) -> builtins.list[MemoryRecord]:
        """Records in the order they were added: skip ``offset``, return up to ``limit``.

        ``limit=None`` returns all the rest.
        """

    def count(self, namespace: str) -> int:
        """The number of records in ``namespace``."""

    def close(self) -> None:
        """Release the storage. Safe to call again."""
