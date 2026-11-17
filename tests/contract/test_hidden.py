"""Hidden records, the same in every store: a record that has expired or is archived stays stored,
reads leave it out unless asked, writes still reach it, and a purge deletes the expired ones."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from typing import Any, NamedTuple

import pytest

from pybodn import Collection, Embedder, MemoryEngine, MemoryRecord, NewMemory
from pybodn._filter import Visibility
from pybodn._storage.base import Store
from pybodn.errors import DuplicateKeyError, RecordNotFoundError, ValidationError
from tests.contract.conftest import LATER, MICROSECOND, START, Clock

MISSING_ID = "0" * 32


class Batch(NamedTuple):
    """One batch of a purge: the moment it went by, and how many records it deleted."""

    now: datetime
    deleted: int


@pytest.fixture
def engine(store: Store, embedder: Embedder | None, clock: Clock) -> MemoryEngine:
    return MemoryEngine(store, clock=clock, embedder=embedder)


def seen(notes: Collection, record: MemoryRecord, **options: Any) -> dict[str, bool]:
    """Whether each read finds ``record``, the only record of ``notes``, with these options."""
    return {
        "get": notes.get(record.id, **options) is not None,
        "get_by_key": record.key is not None
        and notes.get_by_key(record.key, **options) is not None,
        "list": [found.id for found in notes.list(**options)] == [record.id],
        "count": notes.count(**options) == 1,
        "search": [hit.record.id for hit in notes.search("кава", **options)] == [record.id],
    }


EVERYWHERE = dict.fromkeys(["get", "get_by_key", "list", "count", "search"], True)
NOWHERE = dict.fromkeys(EVERYWHERE, False)
EVERY_STATE: Visibility = {"include_archived": True, "include_expired": True}


# expiry --------------------------------------------------------------------------------------


def test_a_record_is_seen_until_the_moment_it_expires(notes: Collection, clock: Clock) -> None:
    # Arrange
    record = notes.add("кава", key="drink", expires_at=LATER)

    # Act
    clock.now = LATER - MICROSECOND
    before = seen(notes, record)
    clock.now = LATER
    at = seen(notes, record)

    # Assert
    assert before == EVERYWHERE
    assert at == NOWHERE


def test_an_expired_record_is_seen_when_asked_for(notes: Collection, clock: Clock) -> None:
    # Arrange
    record = notes.add("кава", key="drink", expires_at=LATER)
    clock.now = LATER

    # Act / Assert
    assert seen(notes, record, include_expired=True) == EVERYWHERE
    assert seen(notes, record, include_archived=True) == NOWHERE


def test_a_moment_already_past_makes_the_record_expired_at_once(notes: Collection) -> None:
    # Act
    record = notes.add("кава", key="drink", expires_at=START - timedelta(days=1))

    # Assert
    assert record.expires_at == START - timedelta(days=1)
    assert seen(notes, record) == NOWHERE


def test_an_expiry_is_kept_as_the_same_moment_in_utc(notes: Collection) -> None:
    # Arrange
    kyiv = timezone(timedelta(hours=2))

    # Act
    record = notes.add("кава", expires_at=datetime(2026, 11, 13, 12, 0, 0, 7, tzinfo=kyiv))

    # Assert
    assert record.expires_at == datetime(2026, 11, 13, 10, 0, 0, 7, tzinfo=UTC)
    assert record.expires_at is not None and record.expires_at.tzinfo is UTC
    assert notes.get(record.id, include_expired=True) == record


def test_a_new_record_never_expires_and_is_not_archived(notes: Collection) -> None:
    # Act
    records = [
        notes.add("кава"),
        *notes.add_many([NewMemory("чай")]),
        notes.put("drink", "молоко"),
    ]

    # Assert
    assert [(record.expires_at, record.archived_at) for record in records] == [(None, None)] * 3


def test_every_way_of_adding_takes_an_expiry(notes: Collection) -> None:
    # Act
    records = [
        notes.add("кава", expires_at=LATER),
        *notes.add_many([NewMemory("чай", expires_at=LATER)]),
        notes.put("drink", "молоко", expires_at=LATER),
    ]

    # Assert
    assert [record.expires_at for record in records] == [LATER] * 3


def test_update_extends_or_clears_the_expiry_even_once_it_has_passed(
    notes: Collection, clock: Clock
) -> None:
    # Arrange
    record = notes.add("кава", key="drink", expires_at=LATER)
    clock.now = LATER

    # Act
    extended = notes.update(record.id, expires_at=LATER + timedelta(hours=1))
    seen_extended = seen(notes, record)
    cleared = notes.update(record.id, expires_at=None)

    # Assert
    assert extended == replace(
        record, expires_at=LATER + timedelta(hours=1), updated_at=LATER, version=2
    )
    assert seen_extended == EVERYWHERE
    assert cleared.expires_at is None


def test_put_replaces_the_expiry_with_the_one_passed(notes: Collection, clock: Clock) -> None:
    # Arrange
    record = notes.put("drink", "кава", expires_at=LATER)
    clock.now = LATER

    # Act
    replaced = notes.put("drink", "кава")

    # Assert
    assert replaced.id == record.id
    assert replaced.expires_at is None
    assert seen(notes, replaced) == EVERYWHERE


# archive -------------------------------------------------------------------------------------


def test_an_archived_record_is_hidden_until_it_is_restored(notes: Collection) -> None:
    # Arrange
    record = notes.add("кава", key="drink")

    # Act
    notes.archive(record.id)
    archived = seen(notes, record), seen(notes, record, include_archived=True)
    notes.restore(record.id)
    restored = seen(notes, record)

    # Assert
    assert archived == (NOWHERE, EVERYWHERE)
    assert seen(notes, record, include_expired=True) == EVERYWHERE  # restored: nothing to ask for
    assert restored == EVERYWHERE


def test_archive_and_restore_each_change_the_record_once(notes: Collection, clock: Clock) -> None:
    # Arrange
    record = notes.add("кава")

    # Act
    clock.now = LATER
    archived = notes.archive(record.id)
    clock.now = LATER + timedelta(hours=1)
    archived_again = notes.archive(record.id)
    restored = notes.restore(record.id)
    clock.now = LATER + timedelta(hours=2)
    restored_again = notes.restore(record.id)

    # Assert
    assert archived == replace(record, archived_at=LATER, updated_at=LATER, version=2)
    assert archived_again == archived
    assert restored == replace(
        archived, archived_at=None, updated_at=LATER + timedelta(hours=1), version=3
    )
    assert restored_again == restored
    assert notes.get(record.id) == restored


@pytest.mark.parametrize("method", ["archive", "restore"])
def test_archive_or_restore_of_a_missing_record_raises(
    engine: MemoryEngine, notes: Collection, method: str
) -> None:
    # Arrange
    elsewhere = engine.collection("user:456").add("кава")

    # Act / Assert
    for record_id in (MISSING_ID, elsewhere.id):
        with pytest.raises(RecordNotFoundError, match=record_id):
            getattr(notes, method)(record_id)


def test_a_record_both_archived_and_expired_is_seen_only_when_both_are_asked_for(
    notes: Collection, clock: Clock
) -> None:
    # Arrange
    record = notes.add("кава", key="drink", expires_at=LATER)
    notes.archive(record.id)
    clock.now = LATER

    # Act / Assert
    assert seen(notes, record, include_archived=True) == NOWHERE
    assert seen(notes, record, include_expired=True) == NOWHERE
    assert seen(notes, record, include_archived=True, include_expired=True) == EVERYWHERE


def test_put_brings_a_record_back_from_the_archive(notes: Collection) -> None:
    # Arrange
    record = notes.put("drink", "кава")
    notes.archive(record.id)

    # Act
    replaced = notes.put("drink", "кава з молоком")

    # Assert
    assert replaced.id == record.id
    assert replaced.archived_at is None
    assert seen(notes, replaced) == EVERYWHERE


def test_update_leaves_a_record_in_the_archive(notes: Collection) -> None:
    # Arrange
    record = notes.add("кава", key="drink")
    archived = notes.archive(record.id)

    # Act
    updated = notes.update(record.id, content="кава з молоком")

    # Assert
    assert updated.archived_at == archived.archived_at
    assert seen(notes, updated) == NOWHERE


# writes reach hidden records -----------------------------------------------------------------


def hide(notes: Collection, clock: Clock, record: MemoryRecord, how: str) -> None:
    if how == "archive":
        notes.archive(record.id)
    else:
        clock.now = LATER


@pytest.mark.parametrize("how", ["archive", "expire"])
def test_a_hidden_record_keeps_its_key_until_it_is_gone(
    notes: Collection, clock: Clock, how: str
) -> None:
    # Arrange
    record = notes.add("кава", key="drink", expires_at=LATER)
    hide(notes, clock, record, how)

    # Act / Assert
    with pytest.raises(DuplicateKeyError, match="archived or expired"):
        notes.add("чай", key="drink")
    with pytest.raises(DuplicateKeyError, match="archived or expired"):
        notes.add_many([NewMemory("чай", key="drink")])
    replaced = notes.put("drink", "чай")
    assert replaced.id == record.id


@pytest.mark.parametrize("how", ["archive", "expire"])
def test_writes_by_id_reach_a_hidden_record(notes: Collection, clock: Clock, how: str) -> None:
    # Arrange
    record = notes.add("кава", expires_at=LATER)
    hide(notes, clock, record, how)

    # Act
    updated = notes.update(record.id, tags=["drink"])
    deleted = notes.delete(record.id)

    # Assert
    assert updated.tags == ("drink",)
    assert deleted is True
    assert notes.get(record.id, include_archived=True, include_expired=True) is None


def test_an_expired_record_can_still_be_archived_and_restored(
    notes: Collection, clock: Clock
) -> None:
    # Arrange
    record = notes.add("кава", key="drink", expires_at=LATER)
    clock.now = LATER

    # Act
    archived = notes.archive(record.id)
    restored = notes.restore(record.id)

    # Assert
    assert (archived.version, restored.version) == (2, 3)
    assert seen(notes, restored) == NOWHERE
    assert seen(notes, restored, include_expired=True) == EVERYWHERE


def test_reads_page_and_filter_only_the_records_they_show(notes: Collection, clock: Clock) -> None:
    # Arrange: of four records, the first is archived and the second expired
    records = [
        notes.add(f"кава {n}", tags=["drink"], expires_at=LATER if n == 1 else None)
        for n in range(4)
    ]
    notes.archive(records[0].id)
    clock.now = LATER
    shown = records[2:]

    # Act / Assert
    assert notes.list(limit=1, offset=1) == [shown[1]]
    assert notes.list(limit=1, offset=1, include_expired=True) == [shown[0]]
    assert notes.list(tags=["drink"], min_importance=0.5) == shown
    assert notes.count(tags=["drink"]) == 2
    assert notes.count(tags=["drink"], **EVERY_STATE) == 4
    assert [hit.record for hit in notes.search("кава", limit=1, mode="text")] == [shown[0]]


def test_hidden_records_still_count_in_the_scores_of_a_search_by_words(
    notes: Collection, clock: Clock
) -> None:
    # Arrange: the word is common, so it counts for less in the score
    kept = notes.add("кава з молоком")
    hidden = [notes.add("кава"), notes.add("кава", expires_at=LATER)]
    [before] = [hit.score for hit in notes.search("кава") if hit.record.id == kept.id]

    # Act
    notes.archive(hidden[0].id)
    clock.now = LATER
    hits = notes.search("кава")

    # Assert
    assert [(hit.record.id, hit.score) for hit in hits] == [(kept.id, before)]


# purge ---------------------------------------------------------------------------------------


def test_purge_deletes_every_expired_record_in_every_namespace(
    engine: MemoryEngine, clock: Clock
) -> None:
    # Arrange
    notes, others = engine.collection("user:1"), engine.collection("user:2")
    expired = [
        notes.add("кава", expires_at=LATER),
        others.add("чай", key="drink", expires_at=LATER),
        notes.add("молоко", expires_at=START),  # expired from the start
    ]
    notes.archive(expired[0].id)
    kept = [
        notes.add("вода"),
        notes.add("сік", expires_at=LATER + MICROSECOND),
        others.archive(others.add("мед").id),
    ]
    clock.now = LATER

    # Act
    purged = engine.purge_expired()

    # Assert
    assert purged == 3
    assert [notes.get(record.id, **EVERY_STATE) for record in expired] == [None] * 3
    assert others.get_by_key("drink", **EVERY_STATE) is None
    assert [record.id for record in [*notes.list(**EVERY_STATE), *others.list(**EVERY_STATE)]] == [
        record.id for record in kept
    ]


def test_purge_with_nothing_expired_deletes_nothing(engine: MemoryEngine, clock: Clock) -> None:
    # Arrange
    notes = engine.collection("user:1")
    records = [notes.add("кава"), notes.add("чай", expires_at=LATER)]

    # Act / Assert
    assert engine.purge_expired() == 0
    assert notes.list(**EVERY_STATE) == records


def test_purge_frees_the_key_of_an_expired_record(engine: MemoryEngine, clock: Clock) -> None:
    # Arrange
    notes = engine.collection("user:1")
    notes.add("кава", key="drink", expires_at=LATER)
    clock.now = LATER

    # Act
    engine.purge_expired()
    added = notes.add("чай", key="drink")

    # Assert
    assert notes.get_by_key("drink") == added


@pytest.fixture
def batches(store: Store, clock: Clock, monkeypatch: pytest.MonkeyPatch) -> list[Batch]:
    """Each batch the store purges, in order. The clock moves on a day after each one."""
    purged: list[Batch] = []
    purge = type(store).purge_expired

    def watched(self: Store, now: datetime, *, limit: int) -> int:
        deleted = purge(self, now, limit=limit)
        purged.append(Batch(now, deleted))
        clock.now += timedelta(days=1)
        return deleted

    monkeypatch.setattr(type(store), "purge_expired", watched)  # the store has no __dict__
    return purged


def test_purge_goes_batch_by_batch(
    engine: MemoryEngine, clock: Clock, batches: list[Batch], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    monkeypatch.setattr("pybodn._engine._PURGE_BATCH", 2)
    notes = engine.collection("user:1")
    notes.add_many([NewMemory(f"запис {n}", expires_at=LATER) for n in range(5)])
    clock.now = LATER

    # Act
    purged = engine.purge_expired()

    # Assert
    assert purged == sum(batch.deleted for batch in batches) == 5
    assert max(batch.deleted for batch in batches) == 2
    assert notes.list(**EVERY_STATE) == []


def test_every_batch_of_a_purge_goes_by_the_moment_it_began(
    engine: MemoryEngine, clock: Clock, batches: list[Batch], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: the clock moves a day on after each batch, past the second expiry
    monkeypatch.setattr("pybodn._engine._PURGE_BATCH", 1)
    notes = engine.collection("user:1")
    notes.add_many(
        [NewMemory("кава", expires_at=LATER), NewMemory("чай", expires_at=LATER + MICROSECOND)]
    )
    clock.now = LATER

    # Act
    purged = engine.purge_expired()

    # Assert
    assert purged == 1
    assert {batch.now for batch in batches} == {LATER}
    assert [record.content for record in notes.list(**EVERY_STATE)] == ["чай"]


# checks --------------------------------------------------------------------------------------

BAD_MOMENTS = [datetime(2026, 11, 13, 12, 0), "2026-11-13T12:00:00+00:00", 0, True]


@pytest.mark.parametrize("value", BAD_MOMENTS)
def test_an_expiry_that_is_not_a_moment_with_a_time_zone_is_refused(
    notes: Collection, value: object
) -> None:
    record = notes.add("кава")
    for write in (
        lambda: notes.add("чай", expires_at=value),  # type: ignore[arg-type]
        lambda: notes.add_many([NewMemory("чай", expires_at=value)]),  # type: ignore[arg-type]
        lambda: notes.put("drink", "чай", expires_at=value),  # type: ignore[arg-type]
        lambda: notes.update(record.id, expires_at=value),  # type: ignore[arg-type]
    ):
        with pytest.raises(ValidationError, match="expires_at"):
            write()
    assert notes.list() == [record]


@pytest.mark.parametrize("option", ["include_archived", "include_expired"])
@pytest.mark.parametrize("value", [1, 0, "yes"])
def test_an_option_that_is_not_true_or_false_is_refused(
    notes: Collection, option: str, value: object
) -> None:
    options: Any = {option: value}
    for read in (
        lambda: notes.get(MISSING_ID, **options),
        lambda: notes.get_by_key("drink", **options),
        lambda: notes.list(**options),
        lambda: notes.count(**options),
        lambda: notes.search("кава", **options),
    ):
        with pytest.raises(ValidationError, match=option):
            read()


def test_an_option_set_to_none_asks_for_nothing(notes: Collection) -> None:
    # Arrange
    record = notes.add("кава", key="drink")
    notes.archive(record.id)

    # Act / Assert
    assert seen(notes, record, include_archived=None, include_expired=None) == NOWHERE
