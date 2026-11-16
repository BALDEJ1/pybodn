"""Mixed search: how the records of the two searches rank as one."""

from dataclasses import replace

from pybodn import SearchHit
from pybodn._mixing import mix
from tests.unit.records import RECORD


def ranking(others: str, length: int, places: dict[int, str]) -> list[SearchHit]:
    """``length`` hits, best first: the records ``places`` names at their places, counted from 1,
    and at every other place a record whose id starts with ``others``."""
    return [
        SearchHit(record=replace(RECORD, id=places.get(place, f"{others}{place}")), score=1.0)
        for place in range(1, length + 1)
    ]


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


def test_equal_scores_keep_the_order_of_the_search_by_words_however_floats_round() -> None:
    # Arrange: "a" stands 12th by words and 28th by meaning, "b" 39th and 6th. Both score
    # exactly 1 / 72 + 1 / 88 = 1 / 99 + 1 / 66, but these sums of floats differ in the last bit.
    by_words = ranking("w", 39, {12: "a", 39: "b"})
    by_meaning = ranking("m", 28, {6: "b", 28: "a"})

    # Act
    hits = [hit for hit in mix(by_words, by_meaning, limit=100) if hit.record.id in {"a", "b"}]

    # Assert
    assert [hit.record.id for hit in hits] == ["a", "b"]
    assert hits[0].score == hits[1].score == 5 / 198
