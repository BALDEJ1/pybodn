"""Mixed search: how the records of the two searches rank as one."""

from dataclasses import replace
from datetime import UTC, datetime

from pybodn import MemoryRecord, SearchHit
from pybodn._mixing import mix

NOW = datetime(2026, 11, 15, 12, 0, tzinfo=UTC)
RECORD = MemoryRecord(
    id="a" * 32,
    namespace="user:123",
    content="кава",
    key=None,
    metadata={},
    tags=(),
    importance=0.5,
    source=None,
    created_at=NOW,
    updated_at=NOW,
    version=1,
)


def test_nothing_found_either_way_mixes_into_nothing() -> None:
    assert mix([], [], limit=10) == []


def test_a_record_both_searches_found_is_as_the_search_by_words_returned_it() -> None:
    # Arrange: its text changed between the two searches
    changed = replace(RECORD, content="чай", version=2)

    # Act
    [hit] = mix(
        [SearchHit(record=RECORD, score=2.5)], [SearchHit(record=changed, score=0.9)], limit=10
    )

    # Assert
    assert hit == SearchHit(record=RECORD, score=1 / 61 + 1 / 61)
