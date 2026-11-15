"""The public API contract. Every test in this folder runs once for each store in ``STORES``."""

import threading
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from pybodn import Collection, Embedder, MemoryEngine
from pybodn._storage.base import Store
from pybodn._storage.memory import InMemoryStore
from pybodn._storage.sqlite import SQLiteStore
from tests.fake_model import FakeModel

# Each one makes a new, empty store; the path is a folder of its own.
STORES: dict[str, Callable[[Path], Store]] = {
    "memory": lambda _: InMemoryStore(),
    "sqlite": lambda folder: SQLiteStore(str(folder / "memory.db"), timeout=5.0),
}

# The name a module of tests gives ``model`` under.
MODEL = "fake-1"

START = datetime(2026, 11, 13, 9, 55, 0, 123456, tzinfo=UTC)
TICK = timedelta(seconds=1, microseconds=1)


class TickingClock:
    """Every reading is a second and a microsecond after the previous one.

    A frozen clock would hide a record whose two times come from two readings, and
    whole seconds would hide a store that drops microseconds.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next = START

    def __call__(self) -> datetime:
        with self._lock:
            now = self._next
            self._next = now + TICK
        return now


def engine_on(store: Store, embedder: Embedder | None = None) -> MemoryEngine:
    """An engine on ``store``, as each process opens the same file. The store outlives it."""
    return MemoryEngine(store, clock=TickingClock(), embedder=embedder)


@pytest.fixture(params=list(STORES.values()), ids=list(STORES))
def store(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[Store]:
    make_store: Callable[[Path], Store] = request.param
    store = make_store(tmp_path)
    yield store
    store.close()


@pytest.fixture
def model() -> FakeModel:
    return FakeModel()


@pytest.fixture
def embedder() -> Embedder | None:
    """The embedding model of ``engine``: none, unless a module of tests gives one."""
    return None


@pytest.fixture
def engine(store: Store, embedder: Embedder | None) -> MemoryEngine:
    return engine_on(store, embedder)


@pytest.fixture
def notes(engine: MemoryEngine) -> Collection:
    return engine.collection("user:123")


@pytest.fixture
def plain(store: Store) -> Collection:
    """The records of ``notes`` through an engine without a model."""
    return engine_on(store).collection("user:123")
