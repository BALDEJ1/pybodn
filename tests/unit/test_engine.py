from datetime import UTC, datetime

import pytest

from pybodn import MemoryEngine
from pybodn.errors import EngineClosedError


def test_an_in_memory_engine_stores_and_reads_back() -> None:
    with MemoryEngine.in_memory() as engine:
        notes = engine.collection("user:123")
        record = notes.add("Prefers Python")
        assert notes.get(record.id) == record


def test_records_get_the_current_utc_time() -> None:
    # Arrange
    before = datetime.now(UTC)

    # Act
    with MemoryEngine.in_memory() as engine:
        record = engine.collection("user:123").add("Prefers Python")
    after = datetime.now(UTC)

    # Assert
    assert record.created_at.tzinfo is UTC
    assert before <= record.created_at <= after


def test_in_memory_engines_do_not_share_records() -> None:
    with MemoryEngine.in_memory() as first, MemoryEngine.in_memory() as second:
        record = first.collection("user:123").add("Prefers Python")
        assert second.collection("user:123").get(record.id) is None


def test_leaving_the_with_block_closes_the_engine() -> None:
    # Act
    with MemoryEngine.in_memory() as engine:
        notes = engine.collection("user:123")

    # Assert
    with pytest.raises(EngineClosedError):
        notes.count()


def test_an_error_in_the_with_block_closes_the_engine_and_propagates() -> None:
    # Arrange
    engine = MemoryEngine.in_memory()
    notes = engine.collection("user:123")

    # Act
    with pytest.raises(RuntimeError, match="boom"), engine:
        raise RuntimeError("boom")

    # Assert
    with pytest.raises(EngineClosedError):
        notes.count()
