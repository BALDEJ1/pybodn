"""Adding with ``dedupe=True``, the same in every store: a memory whose text and key a record the
namespace shows already holds is not written, and that record comes back in its place, as it is."""

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from pybodn import Collection, Embedder, MemoryEngine, NewMemory
from pybodn._filter import EVERYTHING, Visibility
from pybodn._storage.base import Store
from pybodn.errors import DuplicateKeyError, ValidationError
from tests.contract.conftest import LATER, MICROSECOND, START, TICK, Clock, engine_on
from tests.unit.records import RECORD

EVERY_STATE: Visibility = {"include_archived": True, "include_expired": True}
THREADS = 8
WAIT_SECONDS = 10  # a stuck test fails after this instead of hanging


@pytest.fixture
def engine(store: Store, embedder: Embedder | None, clock: Clock) -> MemoryEngine:
    return MemoryEngine(store, clock=clock, embedder=embedder)


# a single memory -------------------------------------------------------------------------------


def test_a_memory_already_there_returns_the_stored_record_as_it_is(notes: Collection) -> None:
    # Arrange
    stored = notes.add("кава", tags=["drink"], metadata={"hot": True}, expires_at=LATER)

    # Act
    again = notes.add("кава", tags=["other"], importance=0.9, source="chat", dedupe=True)

    # Assert
    assert again == stored
    assert notes.list() == [stored]


def test_a_new_text_is_added_as_without_dedupe(notes: Collection) -> None:
    # Arrange
    notes.add("кава")

    # Act
    record = notes.add("чай", key="drink", tags=["hot"], dedupe=True)

    # Assert
    assert (record.content, record.key, record.tags, record.version) == (
        "чай",
        "drink",
        ("hot",),
        1,
    )
    assert notes.count() == 2


@pytest.mark.parametrize(
    "text",
    ["Кава", "кава ", " кава", "кава́", "кава\x00", "ка ва"],
    ids=["case", "space-after", "space-before", "stress-mark", "nul", "space-inside"],
)
def test_only_the_very_same_text_is_a_copy(notes: Collection, text: str) -> None:
    # Arrange
    stored = notes.add("кава")

    # Act
    record = notes.add(text, dedupe=True)

    # Assert
    assert record.id != stored.id
    assert record.content == text
    assert notes.count() == 2


@pytest.mark.parametrize(
    "text",
    ["a\x00b", "😀 емодзі", "\x00", "ї" * 100_000],
    ids=["nul-inside", "emoji", "only-nul", "long"],
)
def test_any_text_finds_its_copy(notes: Collection, text: str) -> None:
    # Arrange
    stored = notes.add(text)
    notes.add(text[:-1] + "x")  # differs in the last character only

    # Act / Assert
    assert notes.add(text, dedupe=True) == stored
    assert notes.count() == 2


def test_a_key_with_a_nul_character_finds_its_copy(notes: Collection) -> None:
    # Arrange
    stored = notes.add("кава", key="a\x00b")
    notes.add("кава", key="a\x00c")

    # Act / Assert
    assert notes.add("кава", key="a\x00b", dedupe=True) == stored


def test_a_batch_with_a_nul_character_finds_every_copy(notes: Collection) -> None:
    # Arrange
    stored = [notes.add(text) for text in ["кава", "😀", "a\x00b"]]

    # Act
    again = notes.add_many([NewMemory(text) for text in ["кава", "😀", "a\x00b"]], dedupe=True)

    # Assert
    assert again == stored


@pytest.mark.parametrize(
    ("stored_key", "key"), [(None, "drink"), ("drink", None), ("drink", "tea")]
)
def test_a_copy_has_the_same_key_too(
    notes: Collection, stored_key: str | None, key: str | None
) -> None:
    # Arrange
    stored = notes.add("кава", key=stored_key)

    # Act
    record = notes.add("кава", key=key, dedupe=True)

    # Assert
    assert record.id != stored.id
    assert record.key == key
    assert notes.count() == 2


def test_a_record_with_the_same_key_and_text_is_a_copy(notes: Collection) -> None:
    # Arrange: the same text without the key, added first
    notes.add("кава")
    stored = notes.add("кава", key="drink")

    # Act / Assert
    assert notes.add("кава", key="drink", dedupe=True) == stored


def test_a_key_used_by_another_text_is_still_refused(notes: Collection) -> None:
    # Arrange
    stored = notes.add("кава", key="drink")

    # Act / Assert
    with pytest.raises(DuplicateKeyError, match="drink"):
        notes.add("чай", key="drink", dedupe=True)
    assert notes.list() == [stored]


def test_the_first_record_added_with_the_text_is_returned(notes: Collection) -> None:
    # Arrange: two copies, added without dedupe
    first = notes.add("кава")
    second = notes.add("кава")

    # Act
    again = notes.add("кава", dedupe=True)
    notes.archive(first.id)
    once_first_is_archived = notes.add("кава", dedupe=True)

    # Assert
    assert again == first
    assert once_first_is_archived == second


def test_a_copy_in_another_namespace_is_no_copy(engine: MemoryEngine, notes: Collection) -> None:
    # Arrange
    theirs = engine.collection("user:456").add("кава")

    # Act
    record = notes.add("кава", dedupe=True)

    # Assert
    assert record.id != theirs.id
    assert record.namespace == "user:123"


def test_the_store_finds_only_the_texts_and_keys_asked_for(store: Store, notes: Collection) -> None:
    # Arrange: records that share a text or a key with what is asked, but never both
    notes.add("кава")
    notes.add("чай", key="tea")
    asked = [replace(RECORD, content="кава", key="tea"), replace(RECORD, content="чай")]

    # Act
    found = store.find_copies("user:123", asked, where=EVERYTHING)

    # Assert
    assert found == {}


# hidden records are no copies ------------------------------------------------------------------


@pytest.mark.parametrize("how", ["archive", "expire"])
def test_a_hidden_record_is_no_copy(notes: Collection, clock: Clock, how: str) -> None:
    # Arrange
    hidden = notes.add("кава", expires_at=LATER)
    if how == "archive":
        notes.archive(hidden.id)
    else:
        clock.now = LATER

    # Act
    record = notes.add("кава", dedupe=True)

    # Assert
    assert record.id != hidden.id
    assert notes.list() == [record]
    assert notes.count(**EVERY_STATE) == 2


@pytest.mark.parametrize("how", ["archive", "expire"])
def test_a_hidden_record_with_the_same_key_still_holds_it(
    notes: Collection, clock: Clock, how: str
) -> None:
    # Arrange
    hidden = notes.add("кава", key="drink", expires_at=LATER)
    if how == "archive":
        notes.archive(hidden.id)
    else:
        clock.now = LATER

    # Act / Assert
    with pytest.raises(DuplicateKeyError, match="archived or expired"):
        notes.add("кава", key="drink", dedupe=True)
    assert notes.count(**EVERY_STATE) == 1


def test_a_record_is_a_copy_until_the_moment_it_expires(notes: Collection, clock: Clock) -> None:
    # Arrange
    stored = notes.add("кава", expires_at=LATER)

    # Act
    clock.now = LATER - MICROSECOND
    before = notes.add("кава", dedupe=True)
    clock.now = LATER
    at = notes.add("кава", dedupe=True)

    # Assert
    assert before == stored
    assert at.id != stored.id


def test_a_copy_is_looked_for_at_the_one_moment_the_call_reads(store: Store) -> None:
    # Arrange: each reading of this engine's clock is a tick after the one before
    notes = engine_on(store).collection("user:123")
    stored = notes.add("кава", expires_at=START + 2 * TICK)  # reads START

    # Act: reads START + TICK, for the new record and for its copies alike
    again = notes.add("кава", dedupe=True)

    # Assert
    assert again == stored


# a batch ---------------------------------------------------------------------------------------


def test_a_repeat_needs_the_same_key_too_within_a_batch(notes: Collection) -> None:
    # Arrange
    stored = notes.add("кава")

    # Act
    records = notes.add_many(
        [
            NewMemory("кава"),
            NewMemory("кава"),
            NewMemory("чай", key="drink"),
            NewMemory("чай"),
        ],
        dedupe=True,
    )

    # Assert
    coffee, coffee_again, tea_with_key, tea = records
    assert coffee == coffee_again == stored
    assert tea_with_key.id != tea.id
    assert notes.list() == [stored, tea_with_key, tea]


def test_a_batch_adds_each_new_text_once_and_returns_a_record_for_each_memory(
    notes: Collection,
) -> None:
    # Arrange
    stored = notes.add("кава")

    # Act
    records = notes.add_many(
        [NewMemory("чай"), NewMemory("кава"), NewMemory("чай"), NewMemory("молоко")], dedupe=True
    )

    # Assert
    tea, coffee, tea_again, milk = records
    assert coffee == stored
    assert tea_again == tea
    assert [record.content for record in records] == ["чай", "кава", "чай", "молоко"]
    assert notes.list() == [stored, tea, milk]


def test_a_repeat_within_a_batch_gets_the_record_of_the_first_such_memory(
    notes: Collection,
) -> None:
    # Act
    first, again = notes.add_many(
        [NewMemory("чай", key="drink", tags=["first"]), NewMemory("чай", key="drink", tags=["2"])],
        dedupe=True,
    )

    # Assert
    assert again == first
    assert first.tags == ("first",)
    assert notes.list() == [first]


def test_without_dedupe_a_batch_writes_every_memory(notes: Collection) -> None:
    # Arrange
    stored = notes.add("чай")

    # Act
    first, again = notes.add_many([NewMemory("чай"), NewMemory("чай")])

    # Assert
    assert len({stored.id, first.id, again.id}) == 3
    with pytest.raises(DuplicateKeyError, match="repeated within the batch"):
        notes.add_many([NewMemory("чай", key="drink"), NewMemory("чай", key="drink")])


def test_two_new_texts_of_a_batch_with_one_key_write_nothing(notes: Collection) -> None:
    with pytest.raises(DuplicateKeyError, match="repeated within the batch"):
        notes.add_many([NewMemory("чай", key="drink"), NewMemory("кава", key="drink")], dedupe=True)
    assert notes.count() == 0


def test_only_the_keys_of_the_memories_written_are_checked(notes: Collection) -> None:
    # Arrange
    stored = notes.add("кава", key="drink")

    # Act
    coffee, tea = notes.add_many([NewMemory("кава", key="drink"), NewMemory("чай")], dedupe=True)

    # Assert
    assert coffee == stored
    assert notes.list() == [stored, tea]


def test_a_batch_with_a_used_key_writes_nothing(notes: Collection) -> None:
    # Arrange
    coffee = notes.add("кава")
    taken = notes.add("молоко", key="drink")

    # Act / Assert
    with pytest.raises(DuplicateKeyError, match="drink"):
        notes.add_many(
            [NewMemory("кава"), NewMemory("чай"), NewMemory("чай", key="drink")], dedupe=True
        )
    assert notes.list() == [coffee, taken]


def test_a_repeat_is_checked_like_any_memory(notes: Collection) -> None:
    # Arrange
    notes.add("кава")

    # Act / Assert
    with pytest.raises(ValidationError, match="importance"):
        notes.add_many([NewMemory("чай"), NewMemory("кава", importance=2)], dedupe=True)
    assert notes.count() == 1


def test_a_batch_of_nothing_with_dedupe_adds_nothing(notes: Collection) -> None:
    assert notes.add_many([], dedupe=True) == []
    assert notes.count() == 0


@pytest.mark.parametrize("value", [1, 0, "yes", None])
def test_dedupe_that_is_not_true_or_false_is_refused(notes: Collection, value: object) -> None:
    with pytest.raises(ValidationError, match="dedupe must be True or False"):
        notes.add("кава", dedupe=value)  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="dedupe must be True or False"):
        notes.add_many([NewMemory("кава")], dedupe=value)  # type: ignore[arg-type]
    assert notes.count() == 0


# at the same time ------------------------------------------------------------------------------


def test_parallel_adds_of_one_text_write_it_once(notes: Collection) -> None:
    # Arrange
    start = threading.Barrier(THREADS, timeout=WAIT_SECONDS)

    def add() -> str:
        start.wait()
        return notes.add("кава", dedupe=True).id

    # Act
    with ThreadPoolExecutor(THREADS) as pool:
        futures = [pool.submit(add) for _ in range(THREADS)]
    ids = {future.result() for future in futures}

    # Assert
    assert len(ids) == 1
    assert notes.count() == 1
