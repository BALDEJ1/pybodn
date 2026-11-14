"""list() and count() keep only the records that pass every filter given."""

from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest

from pybodn import Collection, MemoryEngine, NewMemory

MAX_FILTER_ITEMS = 100  # tags, and metadata fields, in one filter
MAX_INTEGER = 2**63 - 1
MICROSECOND = timedelta(microseconds=1)
PLUS_FIVE = timezone(timedelta(hours=5))


def kept(notes: Collection, **filters: Any) -> list[str]:
    """The content of each record the filters keep. ``count()`` must agree with ``list()``."""
    records = notes.list(**filters)
    assert notes.count(**filters) == len(records)
    return [record.content for record in records]


# tags ----------------------------------------------------------------------------------------


def test_tags_keep_the_records_that_have_every_tag(notes: Collection) -> None:
    # Arrange
    notes.add("both", tags=["health", "food"])
    notes.add("one", tags=["health"])
    notes.add("both-and-more", tags=["work", "food", "health"])
    notes.add("none")

    # Act / Assert
    assert kept(notes, tags=["food", "health"]) == ["both", "both-and-more"]
    assert kept(notes, tags=["health"]) == ["both", "one", "both-and-more"]


def test_no_tags_set_no_condition(notes: Collection) -> None:
    # Arrange
    notes.add("tagged", tags=["health"])
    notes.add("untagged")

    # Act / Assert
    assert kept(notes, tags=[]) == ["tagged", "untagged"]


@pytest.mark.parametrize(
    ("stored", "wanted"),
    [
        pytest.param("Health", "health", id="other-case"),
        pytest.param("health-check", "health", id="longer"),
        pytest.param("health", "healt", id="shorter"),
        pytest.param(" health", "health", id="leading-space"),
        pytest.param("50%Xoff", "50%_off", id="wildcards-are-plain-characters"),
    ],
)
def test_a_tag_matches_only_the_same_whole_tag(notes: Collection, stored: str, wanted: str) -> None:
    # Arrange
    notes.add("other", tags=[stored])

    # Act / Assert
    assert kept(notes, tags=[wanted]) == []


# metadata ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("stored", "wanted", "is_kept"),
    [
        pytest.param("food", "food", True, id="same-text"),
        pytest.param("food", "Food", False, id="other-case"),
        pytest.param("", "", True, id="empty-text"),
        pytest.param(1, 1, True, id="same-integer"),
        pytest.param(1, 1.0, True, id="integer-and-equal-float"),
        pytest.param(1.0, 1, True, id="float-and-equal-integer"),
        pytest.param(2.5, 2.5, True, id="same-float"),
        pytest.param(-0.0, 0, True, id="negative-zero-and-zero"),
        pytest.param(MAX_INTEGER, MAX_INTEGER, True, id="largest-integer"),
        pytest.param(2**53 + 1, float(2**53), False, id="integer-and-nearest-float"),
        pytest.param(True, True, True, id="true"),
        pytest.param(False, False, True, id="false"),
        pytest.param(1, True, False, id="one-is-not-true"),
        pytest.param(True, 1, False, id="true-is-not-one"),
        pytest.param(0, False, False, id="zero-is-not-false"),
        pytest.param(False, 0, False, id="false-is-not-zero"),
        pytest.param(None, None, True, id="null"),
        pytest.param(None, 0, False, id="null-is-not-zero"),
        pytest.param(0, None, False, id="zero-is-not-null"),
        pytest.param("", None, False, id="empty-text-is-not-null"),
        pytest.param("1", 1, False, id="text-is-not-a-number"),
        pytest.param(1, "1", False, id="number-is-not-text"),
        pytest.param("true", True, False, id="text-is-not-true"),
        pytest.param("null", None, False, id="text-is-not-null"),
        pytest.param([1], 1, False, id="list-is-never-equal"),
        pytest.param({"a": 1}, None, False, id="dict-is-never-equal"),
        pytest.param(["a"], '["a"]', False, id="list-is-not-its-json-text"),
        pytest.param({"a": 1}, '{"a":1}', False, id="dict-is-not-its-json-text"),
    ],
)
def test_a_metadata_field_matches_an_equal_json_value(
    notes: Collection, stored: Any, wanted: Any, is_kept: bool
) -> None:
    # Arrange
    notes.add("record", metadata={"field": stored})

    # Act / Assert
    assert kept(notes, metadata={"field": wanted}) == (["record"] if is_kept else [])


def test_none_keeps_only_a_field_that_is_there_and_null(notes: Collection) -> None:
    # Arrange
    notes.add("null", metadata={"field": None})
    notes.add("missing")
    notes.add("other-field", metadata={"other": None})

    # Act / Assert
    assert kept(notes, metadata={"field": None}) == ["null"]


def test_every_metadata_field_given_must_match(notes: Collection) -> None:
    # Arrange
    notes.add("both", metadata={"topic": "food", "level": 2})
    notes.add("topic-only", metadata={"topic": "food", "level": 3})
    notes.add("level-only", metadata={"topic": "work", "level": 2})

    # Act / Assert
    assert kept(notes, metadata={"topic": "food", "level": 2}) == ["both"]


def test_no_metadata_fields_set_no_condition(notes: Collection) -> None:
    # Arrange
    notes.add("with", metadata={"topic": "food"})
    notes.add("without")

    # Act / Assert
    assert kept(notes, metadata={}) == ["with", "without"]


def test_a_metadata_key_is_a_name_not_a_path(notes: Collection) -> None:
    # Arrange
    notes.add("dotted", metadata={"a.b": 1})
    notes.add("nested", metadata={"a": {"b": 1}})

    # Act / Assert
    assert kept(notes, metadata={"a.b": 1}) == ["dotted"]


@pytest.mark.parametrize("key", ['a"b', "a'b", "$", "$.a", "a[0]", "", "ключ", "a b", "%_"])
def test_any_metadata_key_matches_by_its_exact_name(notes: Collection, key: str) -> None:
    # Arrange
    notes.add("named", metadata={key: "value"})
    notes.add("other", metadata={"other": "value"})

    # Act / Assert
    assert kept(notes, metadata={key: "value"}) == ["named"]


# importance ----------------------------------------------------------------------------------


def test_min_importance_keeps_the_records_at_or_above_it(notes: Collection) -> None:
    # Arrange
    for importance in (0.0, 0.2, 0.8, 1.0):
        notes.add(str(importance), importance=importance)

    # Act / Assert
    assert kept(notes, min_importance=0.8) == ["0.8", "1.0"]
    assert kept(notes, min_importance=0) == ["0.0", "0.2", "0.8", "1.0"]
    assert kept(notes, min_importance=1) == ["1.0"]


# creation time -------------------------------------------------------------------------------


def test_created_since_includes_its_moment_and_created_before_excludes_it(
    notes: Collection,
) -> None:
    # Arrange
    first, second, third = (notes.add(content) for content in ("a", "b", "c"))

    # Act / Assert
    assert kept(notes, created_since=second.created_at) == ["b", "c"]
    assert kept(notes, created_before=second.created_at) == ["a"]
    assert kept(notes, created_since=first.created_at, created_before=third.created_at) == [
        "a",
        "b",
    ]


def test_creation_times_are_compared_to_the_microsecond(notes: Collection) -> None:
    # Arrange
    notes.add("a")
    second = notes.add("b")

    # Act / Assert
    assert kept(notes, created_since=second.created_at + MICROSECOND) == []
    assert kept(notes, created_before=second.created_at + MICROSECOND) == ["a", "b"]


def test_a_time_in_another_zone_means_the_same_moment(notes: Collection) -> None:
    # Arrange
    notes.add("a")
    second = notes.add("b")

    # Act / Assert
    assert kept(notes, created_since=second.created_at.astimezone(PLUS_FIVE)) == ["b"]


def test_since_after_before_keeps_nothing(notes: Collection) -> None:
    # Arrange
    record = notes.add("a")

    # Act / Assert
    assert kept(notes, created_since=record.created_at, created_before=record.created_at) == []


def test_the_earliest_and_latest_times_are_accepted(notes: Collection) -> None:
    # Arrange
    notes.add("a")

    # Act / Assert
    assert kept(notes, created_since=datetime.min.replace(tzinfo=UTC)) == ["a"]
    assert kept(notes, created_before=datetime.max.replace(tzinfo=UTC)) == ["a"]


def test_a_batch_shares_one_creation_time(notes: Collection) -> None:
    # Arrange
    notes.add("before")
    batch = notes.add_many([NewMemory("x"), NewMemory("y")])

    # Act / Assert
    assert kept(notes, created_since=batch[0].created_at) == ["x", "y"]


# together ------------------------------------------------------------------------------------


def test_every_filter_given_must_pass(notes: Collection) -> None:
    # Arrange
    since = notes.add("too-early", tags=["health"], metadata={"topic": "food"}).created_at
    notes.add("all", tags=["health"], metadata={"topic": "food"}, importance=0.9)
    notes.add("no-tag", metadata={"topic": "food"}, importance=0.9)
    notes.add("other-topic", tags=["health"], metadata={"topic": "work"}, importance=0.9)
    notes.add("unimportant", tags=["health"], metadata={"topic": "food"}, importance=0.1)

    # Act
    result = kept(
        notes,
        tags=["health"],
        metadata={"topic": "food"},
        min_importance=0.5,
        created_since=since + MICROSECOND,
    )

    # Assert
    assert result == ["all"]


def test_limit_and_offset_page_through_the_kept_records(notes: Collection) -> None:
    # Arrange
    for number in range(7):
        notes.add(str(number), tags=["even"] if number % 2 == 0 else [])

    # Act / Assert
    assert [r.content for r in notes.list(tags=["even"], limit=2)] == ["0", "2"]
    assert [r.content for r in notes.list(tags=["even"], limit=2, offset=2)] == ["4", "6"]
    assert [r.content for r in notes.list(tags=["even"], limit=2, offset=3)] == ["6"]
    assert [r.content for r in notes.list(tags=["even"], offset=4)] == []
    assert [r.content for r in notes.list(tags=["even"], limit=MAX_INTEGER, offset=1)] == [
        "2",
        "4",
        "6",
    ]


def test_filters_never_reach_into_another_namespace(engine: MemoryEngine) -> None:
    # Arrange
    mine = engine.collection("user:1")
    theirs = engine.collection("user:2")
    theirs.add("theirs", tags=["health"], metadata={"topic": "food"})
    mine.add("mine", tags=["health"], metadata={"topic": "food"})

    # Act / Assert
    assert kept(mine, tags=["health"]) == ["mine"]
    assert kept(mine, metadata={"topic": "food"}) == ["mine"]


def test_filters_see_each_record_as_it_is_now(notes: Collection) -> None:
    # Arrange
    changed = notes.add("changed", tags=["todo"], metadata={"state": "open"})
    deleted = notes.add("deleted", tags=["todo"])
    notes.update(changed.id, tags=["done"], metadata={"state": "closed"})
    notes.delete(deleted.id)

    # Act / Assert
    assert kept(notes, tags=["todo"]) == []
    assert kept(notes, tags=["done"], metadata={"state": "closed"}) == ["changed"]


def test_the_largest_filter_works(notes: Collection) -> None:
    # Arrange
    tags = [f"t{number}" for number in range(MAX_FILTER_ITEMS)]
    metadata = {f"k{number}": number for number in range(MAX_FILTER_ITEMS)}
    record = notes.add("all", tags=tags, metadata=metadata, importance=1.0)
    notes.add("one-tag-short", tags=tags[1:], metadata=metadata, importance=1.0)

    # Act
    result = kept(
        notes,
        tags=tags,
        metadata=metadata,
        min_importance=1,
        created_since=record.created_at,
        created_before=record.created_at + MICROSECOND,
    )

    # Assert
    assert result == ["all"]
