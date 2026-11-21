"""The memory engine: the owner of a storage and the way into its collections."""

import os
from datetime import UTC, datetime
from functools import partial
from types import ModuleType
from typing import Self

from pybodn import _validation as validate
from pybodn._collection import Clock, Collection, Vectorize
from pybodn._models import Embedder
from pybodn._storage.base import Store, ensure_open
from pybodn._storage.memory import InMemoryStore
from pybodn._storage.sqlite import SQLiteStore
from pybodn.errors import ConfigurationError

# Records a purge deletes in one transaction: few enough that other writers never wait long.
_PURGE_BATCH = 1_000


def _utc_now() -> datetime:
    return datetime.now(UTC)


class MemoryEngine:
    """Stores memories.

    Create it with ``MemoryEngine.sqlite(path)`` or ``MemoryEngine.in_memory()``, and use it in
    ``with``. The engine owns its storage: ``close()`` closes it, and any use after that raises
    ``EngineClosedError``. Every method of the engine and of its collections raises
    ``ValidationError`` for an invalid argument, ``TypeError`` for an unknown keyword argument,
    and ``StorageError`` if the storage fails.
    """

    __slots__ = ("_clock", "_embedder", "_store", "_vectorize")

    def __init__(
        self, store: Store, *, clock: Clock = _utc_now, embedder: Embedder | None = None
    ) -> None:
        """Internal: use ``MemoryEngine.sqlite()`` or ``MemoryEngine.in_memory()``, which check
        ``embedder`` before they open the store."""
        self._store = store
        self._clock = clock
        self._embedder = embedder
        self._vectorize: Vectorize | None = (
            None if embedder is None else partial(_vectors_module().embed, embedder)
        )

    @classmethod
    def sqlite(
        cls,
        path: str | os.PathLike[str],
        *,
        timeout: float = 5.0,
        embedder: Embedder | None = None,
    ) -> Self:
        """An engine that keeps records in the SQLite file at ``path``, made if missing.

        ``path`` is a string or path object, and ``":memory:"`` keeps records in memory until the
        engine is closed. While another write holds the file, a write waits up to ``timeout``
        seconds, then raises ``StorageError``. ``embedder`` is the embedding model that search by
        meaning uses; it needs NumPy, or this raises ``ConfigurationError``.

        A file from an older pybodn is brought up to date as it opens. A file that cannot be opened,
        is not pybodn's or is in a newer file format, or whose SQLite lacks full-text search or
        JSON, raises ``StorageError`` and stays as it was.
        """
        checked_path = validate.path(path)
        checked_timeout = validate.timeout(timeout)
        checked_embedder = _checked(embedder)
        return cls(SQLiteStore(checked_path, timeout=checked_timeout), embedder=checked_embedder)

    @classmethod
    def in_memory(cls, *, embedder: Embedder | None = None) -> Self:
        """An engine that keeps records in this process's memory until it is closed.

        ``embedder`` is the embedding model that search by meaning uses; it needs NumPy, or this
        raises ``ConfigurationError``.
        """
        checked_embedder = _checked(embedder)
        return cls(InMemoryStore(), embedder=checked_embedder)

    def collection(self, namespace: str) -> Collection:
        """The records of ``namespace``, a non-empty string.

        Namespaces never see each other's records.
        """
        ensure_open(self._store)
        return Collection(
            self._store, validate.text("namespace", namespace), self._clock, self._vectorize
        )

    def embed_missing(self) -> int:
        """Give a vector of the engine's model to every record without one, and return how many.

        It covers every namespace. Records get no vector when an engine without the model writes
        them, and the vectors of another model are replaced: this is how to move to a new model.
        Records go to the model ``batch_size`` at a time, in the order added, up to the last record
        there was at the start, and each batch is written as soon as the model answers: if the model
        fails, the batches before stay written. A record whose text changes meanwhile keeps no
        vector of its old text, and waits for the next call.

        Raises ``ConfigurationError`` if the engine has no model.
        """
        ensure_open(self._store)
        embedder, vectorize = self._embedder, self._vectorize
        if embedder is None or vectorize is None:  # both or neither
            raise ConfigurationError("embed_missing needs an embedding model; this engine has none")
        until = self._store.last_seq()
        after, written = 0, 0
        while missing := self._store.missing_vectors(
            embedder.name, after=after, until=until, limit=embedder.batch_size
        ):
            vectors = vectorize([record.text for record in missing])
            written += self._store.write_vectors([*zip(missing, vectors, strict=True)])
            after = missing[-1].seq
        return written

    def purge_expired(self) -> int:
        """Delete every record that has expired, in every namespace, and return how many.

        Expired records stay in the store, hidden, until this deletes them, and their data stays
        in the file until ``erase_deleted()``. They go in batches, each deleted at once: if one
        fails, the batches before stay deleted. Every batch goes by the moment the call began, so
        a record that expires meanwhile waits for the next call.
        """
        now = self._clock()
        purged = 0
        while deleted := self._store.purge_expired(now, limit=_PURGE_BATCH):
            purged += deleted
        return purged

    def erase_deleted(self) -> None:
        """Erase deleted data from the file and its log.

        A delete, a ``put`` or ``update`` that replaces values, and ``purge_expired`` remove data
        from every read at once, but it stays in the file until this erases what they removed before
        the call. Then a program reading the file itself no longer finds it. Archived records and
        expired ones not yet purged are still stored, and stay. It needs free disk space up to twice
        the size of the file, and other writers wait for it up to their timeout, then raise
        ``StorageError``. If another connection keeps the file busy all the while, or the file
        cannot be written, it raises ``StorageError``: the file is intact, deleted data may still be
        in it, and the call can be made again later. In memory, there is nothing to erase.
        """
        self._store.erase_deleted()

    def close(self) -> None:
        """Close the storage. Safe to call again."""
        self._store.close()

    def __enter__(self) -> Self:
        ensure_open(self._store)
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _checked(embedder: object) -> Embedder | None:
    """``embedder`` once checked, and NumPy with it, so that a bad one fails before a store
    opens."""
    checked = validate.embedder(embedder)
    if checked is not None:
        _vectors_module()
    return checked


def _vectors_module() -> ModuleType:
    """The module of search by meaning. It needs NumPy, and so is imported only with a model.

    A NumPy that is missing raises ``ConfigurationError`` with the command that installs it; one
    that is there but fails to import raises its own error.
    """
    try:
        from pybodn import _vectors  # noqa: PLC0415
    except ModuleNotFoundError as error:
        raise ConfigurationError(
            'search by meaning needs NumPy: install it with pip install "pybodn[vector]"'
        ) from error
    return _vectors
