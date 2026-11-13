import sys
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor

import pytest

from pybodn import Collection, MemoryEngine
from pybodn.errors import EngineClosedError

THREADS = 8
CALLS = 50
WAIT_SECONDS = 10  # a stuck test fails after this instead of hanging
SWITCH_SECONDS = 1e-6


@pytest.fixture(autouse=True)
def frequent_thread_switches() -> Iterator[None]:
    """Switch threads as often as Python allows, so a missing lock shows in every run."""
    previous = sys.getswitchinterval()
    sys.setswitchinterval(SWITCH_SECONDS)
    yield
    sys.setswitchinterval(previous)


def run_together(work: Callable[[], object]) -> None:
    """Run ``work`` in ``THREADS`` threads; re-raise the first failure."""
    with ThreadPoolExecutor(THREADS) as pool:
        futures = [pool.submit(work) for _ in range(THREADS)]
    for future in futures:
        future.result()


def test_parallel_puts_of_one_key_lose_nothing(notes: Collection) -> None:
    # Arrange
    start = threading.Barrier(THREADS, timeout=WAIT_SECONDS)

    def put_many() -> None:
        start.wait()
        for _ in range(CALLS):
            notes.put("counter", "value")

    # Act
    run_together(put_many)

    # Assert
    record = notes.get_by_key("counter")
    assert record is not None
    assert record.version == THREADS * CALLS
    assert notes.count() == 1


def test_parallel_updates_of_one_record_lose_nothing(notes: Collection) -> None:
    # Arrange
    record = notes.add("value")
    start = threading.Barrier(THREADS, timeout=WAIT_SECONDS)

    def update_many() -> None:
        start.wait()
        for _ in range(CALLS):
            notes.update(record.id, content="value")

    # Act
    run_together(update_many)

    # Assert
    stored = notes.get(record.id)
    assert stored is not None
    assert stored.version == 1 + THREADS * CALLS


def test_actions_racing_close_end_with_engine_closed_error(engine: MemoryEngine) -> None:
    # Arrange
    notes = engine.collection("user:123")
    running = threading.Barrier(THREADS + 1, timeout=WAIT_SECONDS)

    def add_until_closed() -> bool:
        notes.add("value")
        running.wait()
        deadline = time.monotonic() + WAIT_SECONDS
        while time.monotonic() < deadline:
            try:
                notes.add("value")
            except EngineClosedError:
                return True
        return False

    # Act
    with ThreadPoolExecutor(THREADS) as pool:
        futures = [pool.submit(add_until_closed) for _ in range(THREADS)]
        running.wait()
        engine.close()

    # Assert
    assert [future.result() for future in futures] == [True] * THREADS
