"""The async engine: each method is the sync method of the same name, run in one of a few threads
the event loop has for its engines."""

import asyncio
import contextvars
import gc
import inspect
import threading
import weakref
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import create_autospec

import pytest

from pybodn import AsyncCollection, AsyncMemoryEngine, Collection, MemoryEngine, NewMemory
from pybodn._async import _MAX_CALLS_AT_ONCE
from pybodn._storage.memory import InMemoryStore
from pybodn.errors import EngineClosedError, ValidationError

NOW = datetime(2026, 11, 17, 12, 0, tzinfo=UTC)
WAIT_SECONDS = 10  # a stuck test fails after this instead of hanging
SHORT_WAIT = 0.05  # seconds: far more than a free call takes
RESULT = object()  # what a sync method gives back, to see that the async one returns it

# Each async class, with the sync class it mirrors.
TWINS = [(MemoryEngine, AsyncMemoryEngine), (Collection, AsyncCollection)]


def public(cls: type) -> dict[str, Any]:
    return {name: getattr(cls, name) for name in dir(cls) if not name.startswith("_")}


def engine_with(clock: Any) -> AsyncMemoryEngine:
    """An async engine on a store in memory, whose clock is ``clock``: every read calls it."""
    return AsyncMemoryEngine(MemoryEngine(InMemoryStore(), clock=clock))


class Gate:
    """A clock that holds every call reading it until opened, and counts the calls it holds."""

    def __init__(self) -> None:
        self._held = threading.Semaphore(0)
        self._opened = threading.Event()

    def __call__(self) -> datetime:
        self._held.release()
        assert self._opened.wait(WAIT_SECONDS), "the gate was never opened"
        return NOW

    async def holds(self, calls: int) -> None:
        """Wait until ``calls`` more calls are held."""
        for _ in range(calls):
            assert await asyncio.to_thread(self._held.acquire, timeout=WAIT_SECONDS)

    async def one_more_comes(self) -> bool:
        """Whether one more call comes within a wait far longer than a free call takes."""
        await asyncio.sleep(SHORT_WAIT)
        return self._held.acquire(blocking=False)

    def open(self) -> None:
        self._opened.set()


# The twins -----------------------------------------------------------------------------------


@pytest.mark.parametrize(("sync", "twin"), TWINS, ids=["engine", "collection"])
def test_an_async_class_has_the_public_names_of_its_sync_class(sync: type, twin: type) -> None:
    assert sorted(public(twin)) == sorted(public(sync))


@pytest.mark.parametrize(
    ("sync", "twin", "name"),
    [pytest.param(sync, twin, name, id=name) for sync, twin in TWINS for name in public(sync)],
)
def test_an_async_method_takes_what_its_sync_method_takes(
    sync: type, twin: type, name: str
) -> None:
    # Arrange: a property is compared by what reads it
    mine, theirs = public(sync)[name], public(twin)[name]
    is_property = isinstance(mine, property)
    if is_property:
        mine, theirs = mine.fget, theirs.fget
    expected = inspect.signature(mine)
    if expected.return_annotation is Collection:
        expected = expected.replace(return_annotation=AsyncCollection)

    # Act
    signature = inspect.signature(theirs)

    # Assert: every method is awaited, but for the one that only hands out a collection
    assert signature == expected
    assert inspect.iscoroutinefunction(theirs) == (not is_property and name != "collection")


# Each awaited method of an engine or a collection, with arguments unlike every default.
CALLS: dict[str, tuple[tuple[object, ...], dict[str, object]]] = {
    "add": (("кава",), {"dedupe": True, "key": "drink", "tags": ["food"]}),
    "add_many": (([NewMemory("кава")],), {"dedupe": True}),
    "put": (("drink", "кава"), {"importance": 0.9}),
    "get": (("id",), {"include_archived": True}),
    "get_by_key": (("drink",), {"include_expired": True}),
    "update": (("id",), {"expected_version": 2, "content": "чай"}),
    "archive": (("id",), {}),
    "restore": (("id",), {}),
    "delete": (("id",), {}),
    "list": ((), {"limit": 1, "offset": 2, "tags": ["food"]}),
    "count": ((), {"min_importance": 0.5}),
    "search": (("кава",), {"limit": 3, "mode": "text"}),
    "embed_missing": ((), {}),
    "purge_expired": ((), {}),
}
TESTED_ON_THEIR_OWN = {"sqlite", "in_memory", "close"}


def test_every_awaited_method_is_called_below() -> None:
    awaited = {
        name
        for _, twin in TWINS
        for name, member in public(twin).items()
        if inspect.iscoroutinefunction(member)
    }
    assert sorted(CALLS) == sorted(awaited - TESTED_ON_THEIR_OWN)


def recording(threads: list[int]) -> Any:
    """A stand-in for a sync method: notes the thread it runs in and returns ``RESULT``."""

    def call(*args: object, **kwargs: object) -> object:
        threads.append(threading.get_ident())
        return RESULT

    return call


@pytest.mark.parametrize("name", CALLS)
def test_an_awaited_method_hands_its_arguments_to_the_sync_one_in_another_thread(
    name: str,
) -> None:
    # Arrange: a sync engine and collection that note the thread each call runs in
    engine = create_autospec(MemoryEngine, instance=True)
    collection = engine.collection.return_value = create_autospec(Collection, instance=True)
    of_engine = hasattr(MemoryEngine, name)
    sync = getattr(engine if of_engine else collection, name)
    threads: list[int] = []
    sync.side_effect = recording(threads)
    twin = AsyncMemoryEngine(engine)
    args, kwargs = CALLS[name]

    # Act
    method = getattr(twin if of_engine else twin.collection("user:123"), name)
    result = asyncio.run(method(*args, **kwargs))

    # Assert
    assert result is RESULT
    sync.assert_called_once_with(*args, **kwargs)
    assert len(threads) == 1
    assert threads[0] != threading.get_ident()


def test_a_collection_is_the_sync_engines_collection_of_that_namespace() -> None:
    # Arrange
    engine = create_autospec(MemoryEngine, instance=True)
    engine.collection.return_value.namespace = "user:123"

    # Act
    notes = AsyncMemoryEngine(engine).collection("user:123")

    # Assert
    engine.collection.assert_called_once_with("user:123")
    assert notes.namespace == "user:123"


# Opening -------------------------------------------------------------------------------------


def test_sqlite_opens_the_file_with_the_options_given(tmp_path: Path) -> None:
    # Arrange
    path = tmp_path / "memory.db"

    async def add_and_close() -> None:
        engine = await AsyncMemoryEngine.sqlite(path, timeout=1.0)
        await engine.collection("user:123").add("Любить каву")
        await engine.close()

    # Act
    asyncio.run(add_and_close())

    # Assert
    with MemoryEngine.sqlite(path) as engine:
        assert [record.content for record in engine.collection("user:123").list()] == [
            "Любить каву"
        ]


@pytest.mark.parametrize(
    "opening",
    [
        pytest.param(lambda: AsyncMemoryEngine.sqlite(""), id="sqlite-empty-path"),
        pytest.param(
            lambda: AsyncMemoryEngine.sqlite("memory.db", timeout=-1), id="sqlite-timeout"
        ),
        pytest.param(
            lambda: AsyncMemoryEngine.in_memory(embedder="model"),  # type: ignore[arg-type]
            id="in-memory-embedder",
        ),
    ],
)
def test_opening_checks_its_options_as_the_sync_engine_does(opening: Any) -> None:
    with pytest.raises(ValidationError):
        asyncio.run(opening())


def test_in_memory_opens_an_empty_engine() -> None:
    async def count() -> int:
        async with await AsyncMemoryEngine.in_memory() as engine:
            return await engine.collection("user:123").count()

    assert asyncio.run(count()) == 0


@pytest.mark.parametrize(
    "opening",
    [
        pytest.param(AsyncMemoryEngine.sqlite, id="sqlite"),
        pytest.param(lambda _: AsyncMemoryEngine.in_memory(), id="in-memory"),
    ],
)
def test_opening_waits_for_a_free_thread(opening: Any, tmp_path: Path) -> None:
    gate = Gate()

    async def open_while_every_thread_is_taken() -> bool:
        async with engine_with(gate) as engine:
            # Arrange
            notes = engine.collection("user:123")
            held = [asyncio.create_task(notes.count()) for _ in range(_MAX_CALLS_AT_ONCE)]
            await gate.holds(_MAX_CALLS_AT_ONCE)

            # Act
            opened = asyncio.create_task(opening(tmp_path / "memory.db"))
            await asyncio.sleep(SHORT_WAIT)
            waited = not opened.done()
            gate.open()
            await asyncio.gather(*held)
            await (await opened).close()
            return waited

    # Assert
    assert asyncio.run(open_while_every_thread_is_taken())


# Closing -------------------------------------------------------------------------------------


def test_after_close_every_call_raises_engine_closed_error_and_close_is_safe_again() -> None:
    async def close_then_use() -> None:
        engine = await AsyncMemoryEngine.in_memory()
        notes = engine.collection("user:123")
        await engine.close()
        await engine.close()
        with pytest.raises(EngineClosedError):
            await notes.add("Любить каву")
        with pytest.raises(EngineClosedError):
            await engine.purge_expired()
        with pytest.raises(EngineClosedError):
            engine.collection("user:123")
        with pytest.raises(EngineClosedError):
            async with engine:
                pass

    asyncio.run(close_then_use())


def test_close_does_not_wait_for_a_free_thread() -> None:
    gate = Gate()

    async def close_while_every_thread_is_taken() -> list[object]:
        # Arrange
        engine = engine_with(gate)
        notes = engine.collection("user:123")
        held = [asyncio.create_task(notes.count()) for _ in range(_MAX_CALLS_AT_ONCE)]
        await gate.holds(_MAX_CALLS_AT_ONCE)

        # Act
        await asyncio.wait_for(engine.close(), WAIT_SECONDS)

        # Assert: the calls it held go on to find the engine closed
        with pytest.raises(EngineClosedError):
            engine.collection("user:123")
        gate.open()
        return await asyncio.gather(*held, return_exceptions=True)

    results = asyncio.run(close_while_every_thread_is_taken())
    assert [type(result) for result in results] == [EngineClosedError] * _MAX_CALLS_AT_ONCE


def test_async_with_gives_the_engine_and_closes_it() -> None:
    async def use() -> AsyncMemoryEngine:
        async with await AsyncMemoryEngine.in_memory() as engine:
            await engine.collection("user:123").add("Любить каву")
        return engine

    engine = asyncio.run(use())
    with pytest.raises(EngineClosedError):
        engine.collection("user:123")


# Threads -------------------------------------------------------------------------------------


def test_the_event_loop_runs_on_while_a_call_waits() -> None:
    # Arrange: the call waits for something only the event loop does
    release = threading.Event()

    def clock() -> datetime:
        assert release.wait(WAIT_SECONDS), "the call held the event loop"
        return NOW

    async def count_while_releasing() -> int:
        async with engine_with(clock) as engine:
            waiting = asyncio.create_task(engine.collection("user:123").count())
            await asyncio.sleep(0)  # the call starts
            release.set()
            return await waiting

    # Act / Assert
    assert asyncio.run(count_while_releasing()) == 0


def test_the_engines_on_an_event_loop_run_a_few_calls_at_once_together() -> None:
    gate = Gate()

    async def count_one_past_the_limit_on_two_engines() -> list[int]:
        async with engine_with(gate) as first, engine_with(gate) as second:
            # Act
            calls = [
                asyncio.create_task((first, second)[number % 2].collection("user:123").count())
                for number in range(_MAX_CALLS_AT_ONCE + 1)
            ]

            # Assert: the limit of calls go in, and the next one waits for a free thread
            await gate.holds(_MAX_CALLS_AT_ONCE)
            assert not await gate.one_more_comes()
            gate.open()
            return await asyncio.gather(*calls)

    assert asyncio.run(count_one_past_the_limit_on_two_engines()) == [0] * (_MAX_CALLS_AT_ONCE + 1)


def test_each_event_loop_has_threads_of_its_own() -> None:
    gate = Gate()

    async def count() -> int:
        return await engine_with(lambda: NOW).collection("user:123").count()

    async def count_on_another_event_loop_while_every_thread_is_taken() -> int:
        async with engine_with(gate) as engine:
            # Arrange
            notes = engine.collection("user:123")
            held = [asyncio.create_task(notes.count()) for _ in range(_MAX_CALLS_AT_ONCE)]
            await gate.holds(_MAX_CALLS_AT_ONCE)

            # Act
            counted = await asyncio.to_thread(asyncio.run, count())
            gate.open()
            await asyncio.gather(*held)
            return counted

    # Assert: the other event loop did not wait for this one's threads, which the gate held
    assert asyncio.run(count_on_another_event_loop_while_every_thread_is_taken()) == 0


def test_an_engine_serves_one_event_loop_after_another_and_keeps_none_once_it_ends() -> None:
    # Arrange: more calls than threads, so that some wait their turn on each event loop
    engine = engine_with(lambda: NOW)
    loops: list[weakref.ref[asyncio.AbstractEventLoop]] = []

    async def count_many() -> list[int]:
        loops.append(weakref.ref(asyncio.get_running_loop()))
        notes = engine.collection("user:123")
        return await asyncio.gather(*(notes.count() for _ in range(_MAX_CALLS_AT_ONCE * 2)))

    # Act / Assert
    for _ in range(2):
        assert asyncio.run(count_many()) == [0] * _MAX_CALLS_AT_ONCE * 2
    gc.collect()
    assert [loop() for loop in loops] == [None, None]


def test_a_call_given_up_on_keeps_its_thread_until_it_ends() -> None:
    gate = Gate()

    async def give_up_then_count() -> int:
        async with engine_with(gate) as engine:
            # Arrange: held calls take every thread
            notes = engine.collection("user:123")
            held = [asyncio.create_task(notes.count()) for _ in range(_MAX_CALLS_AT_ONCE)]
            await gate.holds(_MAX_CALLS_AT_ONCE)

            # Act: give up on them, as a timeout would, and call again
            for task in held:
                task.cancel()
            later = asyncio.create_task(notes.count())

            # Assert: the new call waits for a free thread, and gets one once the held calls end
            assert not await gate.one_more_comes()
            gate.open()
            return await later

    assert asyncio.run(give_up_then_count()) == 0


def test_a_call_sees_the_context_variables_of_its_caller() -> None:
    # Arrange
    request = contextvars.ContextVar[str]("request")
    seen: list[str] = []

    def clock() -> datetime:
        seen.append(request.get("none"))
        return NOW

    async def count_in_a_request() -> None:
        async with engine_with(clock) as engine:
            request.set("request-1")
            await engine.collection("user:123").count()

    # Act
    asyncio.run(count_in_a_request())

    # Assert
    assert seen == ["request-1"]
