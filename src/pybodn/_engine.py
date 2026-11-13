"""The memory engine: the owner of a storage and the way into its collections."""

from datetime import UTC, datetime
from typing import Self

from pybodn import _validation as validate
from pybodn._collection import Clock, Collection
from pybodn._storage.base import Store
from pybodn._storage.memory import InMemoryStore
from pybodn.errors import EngineClosedError


def _utc_now() -> datetime:
    return datetime.now(UTC)


class MemoryEngine:
    """Stores memories. Create it with ``MemoryEngine.in_memory()`` and use it in ``with``.

    The engine owns its storage: ``close()`` closes it, and any use after that raises
    ``EngineClosedError``.
    """

    __slots__ = ("_clock", "_store")

    def __init__(self, store: Store, *, clock: Clock = _utc_now) -> None:
        """Internal: use a factory such as ``MemoryEngine.in_memory()``."""
        self._store = store
        self._clock = clock

    @classmethod
    def in_memory(cls) -> Self:
        """An engine that keeps records in this process's memory until it is closed."""
        return cls(InMemoryStore())

    def collection(self, namespace: str) -> Collection:
        """The records of ``namespace``. Namespaces never see each other's records."""
        self._check_open()
        return Collection(self._store, validate.text("namespace", namespace), self._clock)

    def close(self) -> None:
        """Close the storage. Safe to call again."""
        self._store.close()

    def __enter__(self) -> Self:
        self._check_open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _check_open(self) -> None:
        if self._store.closed:
            raise EngineClosedError("the engine is closed")
