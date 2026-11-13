"""The memory engine: the owner of a storage and the way into its collections."""

import os
from datetime import UTC, datetime
from typing import Self

from pybodn import _validation as validate
from pybodn._collection import Clock, Collection
from pybodn._storage.base import Store, ensure_open
from pybodn._storage.memory import InMemoryStore
from pybodn._storage.sqlite import SQLiteStore


def _utc_now() -> datetime:
    return datetime.now(UTC)


class MemoryEngine:
    """Stores memories. Create it with ``MemoryEngine.sqlite(path)`` or
    ``MemoryEngine.in_memory()``, and use it in ``with``.

    The engine owns its storage: ``close()`` closes it, and any use after that raises
    ``EngineClosedError``.
    """

    __slots__ = ("_clock", "_store")

    def __init__(self, store: Store, *, clock: Clock = _utc_now) -> None:
        """Internal: use ``MemoryEngine.sqlite()`` or ``MemoryEngine.in_memory()``."""
        self._store = store
        self._clock = clock

    @classmethod
    def sqlite(cls, path: str | os.PathLike[str], *, timeout: float = 5.0) -> Self:
        """An engine that keeps records in the SQLite file at ``path``, made if missing.

        While another process writes to the file, a write waits up to ``timeout`` seconds,
        then raises ``StorageError``. ``":memory:"`` keeps records in memory until the
        engine is closed.
        """
        checked_path = validate.path(path)
        checked_timeout = validate.timeout(timeout)
        return cls(SQLiteStore(checked_path, timeout=checked_timeout))

    @classmethod
    def in_memory(cls) -> Self:
        """An engine that keeps records in this process's memory until it is closed."""
        return cls(InMemoryStore())

    def collection(self, namespace: str) -> Collection:
        """The records of ``namespace``. Namespaces never see each other's records."""
        ensure_open(self._store)
        return Collection(self._store, validate.text("namespace", namespace), self._clock)

    def close(self) -> None:
        """Close the storage. Safe to call again."""
        self._store.close()

    def __enter__(self) -> Self:
        ensure_open(self._store)
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
