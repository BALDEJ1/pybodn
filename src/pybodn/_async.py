"""The async engine: each method runs the sync method of the same name in a thread.

The checks, errors and results are those of ``MemoryEngine`` and ``Collection``, and the event
loop runs on while a call waits for the file's lock or the embedding model.
"""

import asyncio
import builtins
import contextvars
import functools
import os
import weakref
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Self, Unpack

from pybodn._collection import Collection, MemoryChanges, MemoryFields, NewMemoryFields
from pybodn._engine import MemoryEngine
from pybodn._filter import Filters, Visibility
from pybodn._models import Embedder, MemoryRecord, NewMemory, SearchHit, SearchMode

# Each event loop has this many threads of its own for the calls of all its engines; the calls
# past this many wait their turn. Calls waiting for the file's lock would otherwise fill the
# loop's default pool, which also serves the rest of the program, such as looking up addresses.
# The threads are the loop's own, not the program's, so that a child process, whose event loops
# are new, never waits for threads it does not have; they go when the loop goes.
_MAX_CALLS_AT_ONCE = 4

_threads_of: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, ThreadPoolExecutor] = (
    weakref.WeakKeyDictionary()
)


async def _in_thread[**P, R](function: Callable[P, R], /, *args: P.args, **kwargs: P.kwargs) -> R:
    """``function`` run in a thread of the running event loop's own, with the caller's context.

    A call given up on, as on a timeout, once begun runs on in its thread, which it keeps until
    it ends: a thread cannot be stopped.
    """
    loop = asyncio.get_running_loop()
    threads = _threads_of.get(loop)
    if threads is None:
        threads = _threads_of[loop] = ThreadPoolExecutor(
            _MAX_CALLS_AT_ONCE, thread_name_prefix="pybodn"
        )
    call = functools.partial(contextvars.copy_context().run, function, *args, **kwargs)
    return await loop.run_in_executor(threads, call)


class AsyncCollection:
    """``Collection`` for asyncio. Get it from ``AsyncMemoryEngine.collection()``.

    Each method does what the method of the same name of ``Collection`` does, in a thread.
    """

    __slots__ = ("_collection",)

    def __init__(self, collection: Collection) -> None:
        """Internal: use ``AsyncMemoryEngine.collection()``."""
        self._collection = collection

    @property
    def namespace(self) -> str:
        """The namespace of every record in this collection."""
        return self._collection.namespace

    async def add(
        self, content: str, *, dedupe: bool = False, **fields: Unpack[NewMemoryFields]
    ) -> MemoryRecord:
        """Add a record, as ``Collection.add()`` does."""
        return await _in_thread(self._collection.add, content, dedupe=dedupe, **fields)

    async def add_many(
        self, memories: Iterable[NewMemory], *, dedupe: bool = False
    ) -> builtins.list[MemoryRecord]:
        """Add every memory, or none, as ``Collection.add_many()`` does."""
        return await _in_thread(self._collection.add_many, memories, dedupe=dedupe)

    async def put(self, key: str, content: str, **fields: Unpack[MemoryFields]) -> MemoryRecord:
        """Create or replace the record with ``key``, as ``Collection.put()`` does."""
        return await _in_thread(self._collection.put, key, content, **fields)

    async def get(self, id: str, **options: Unpack[Visibility]) -> MemoryRecord | None:
        """The record with ``id``, or ``None``, as ``Collection.get()`` gives it."""
        return await _in_thread(self._collection.get, id, **options)

    async def get_by_key(self, key: str, **options: Unpack[Visibility]) -> MemoryRecord | None:
        """The record with ``key``, or ``None``, as ``Collection.get_by_key()`` gives it."""
        return await _in_thread(self._collection.get_by_key, key, **options)

    async def update(
        self, id: str, *, expected_version: int | None = None, **changes: Unpack[MemoryChanges]
    ) -> MemoryRecord:
        """Change the fields passed, as ``Collection.update()`` does."""
        return await _in_thread(
            self._collection.update, id, expected_version=expected_version, **changes
        )

    async def archive(self, id: str) -> MemoryRecord:
        """Hide the record from every read, as ``Collection.archive()`` does."""
        return await _in_thread(self._collection.archive, id)

    async def restore(self, id: str) -> MemoryRecord:
        """Bring the record back from the archive, as ``Collection.restore()`` does."""
        return await _in_thread(self._collection.restore, id)

    async def delete(self, id: str) -> bool:
        """Remove the record, as ``Collection.delete()`` does."""
        return await _in_thread(self._collection.delete, id)

    async def list(
        self, *, limit: int | None = None, offset: int = 0, **filters: Unpack[Filters]
    ) -> builtins.list[MemoryRecord]:
        """Records in the order they were added, as ``Collection.list()`` gives them."""
        return await _in_thread(self._collection.list, limit=limit, offset=offset, **filters)

    async def count(self, **filters: Unpack[Filters]) -> int:
        """The number of records, as ``Collection.count()`` gives it."""
        return await _in_thread(self._collection.count, **filters)

    async def search(
        self,
        query: str,
        *,
        limit: int = 10,
        mode: SearchMode = "auto",
        **filters: Unpack[Filters],
    ) -> builtins.list[SearchHit]:
        """The records that best match ``query``, as ``Collection.search()`` finds them."""
        return await _in_thread(self._collection.search, query, limit=limit, mode=mode, **filters)


class AsyncMemoryEngine:
    """``MemoryEngine`` for asyncio: open it with ``await AsyncMemoryEngine.sqlite(path)`` or
    ``await AsyncMemoryEngine.in_memory()``, and use it in ``async with``.

    Each method does what the method of the same name of ``MemoryEngine`` does, in one of a few
    threads the event loop has for all its engines. A call given up on, as on a timeout, once
    begun still runs to its end, so a write may still land.
    """

    __slots__ = ("_engine",)

    def __init__(self, engine: MemoryEngine) -> None:
        """Internal: use ``AsyncMemoryEngine.sqlite()`` or ``AsyncMemoryEngine.in_memory()``."""
        self._engine = engine

    @classmethod
    async def sqlite(
        cls,
        path: str | os.PathLike[str],
        *,
        timeout: float = 5.0,
        embedder: Embedder | None = None,
    ) -> Self:
        """``MemoryEngine.sqlite()``, opened in a thread."""
        return cls(await _in_thread(MemoryEngine.sqlite, path, timeout=timeout, embedder=embedder))

    @classmethod
    async def in_memory(cls, *, embedder: Embedder | None = None) -> Self:
        """``MemoryEngine.in_memory()``, opened in a thread."""
        return cls(await _in_thread(MemoryEngine.in_memory, embedder=embedder))

    def collection(self, namespace: str) -> AsyncCollection:
        """The records of ``namespace``, as ``MemoryEngine.collection()`` gives them."""
        return AsyncCollection(self._engine.collection(namespace))

    async def embed_missing(self) -> int:
        """Give a vector to every record that has none, as ``MemoryEngine.embed_missing()`` does."""
        return await _in_thread(self._engine.embed_missing)

    async def purge_expired(self) -> int:
        """Delete every record that has expired, as ``MemoryEngine.purge_expired()`` does."""
        return await _in_thread(self._engine.purge_expired)

    async def erase_deleted(self) -> None:
        """Erase hidden data from the file, as ``MemoryEngine.erase_deleted()`` does."""
        return await _in_thread(self._engine.erase_deleted)

    async def close(self) -> None:
        """Close the storage. Safe to call again.

        It does not wait for a thread of the event loop's engines, and once begun it runs to its
        end, even if given up on. The store still waits for the call it is making.
        """
        await asyncio.to_thread(self._engine.close)

    async def __aenter__(self) -> Self:
        self._engine.__enter__()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()
