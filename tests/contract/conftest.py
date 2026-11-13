"""The public API contract. Every test in this folder runs once for each store in ``STORES``."""

import threading
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from pybodn import Collection, MemoryEngine
from pybodn._storage.base import Store
from pybodn._storage.memory import InMemoryStore

# SQLite joins this list in step 3.
STORES: dict[str, Callable[[Path], Store]] = {
    "memory": lambda _: InMemoryStore(),
}

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


@pytest.fixture(params=list(STORES.values()), ids=list(STORES))
def engine(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[MemoryEngine]:
    make_store: Callable[[Path], Store] = request.param
    with MemoryEngine(make_store(tmp_path), clock=TickingClock()) as engine:
        yield engine


@pytest.fixture
def notes(engine: MemoryEngine) -> Collection:
    return engine.collection("user:123")
