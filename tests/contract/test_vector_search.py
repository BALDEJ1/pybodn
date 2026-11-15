"""Search by meaning, the same in every store: what an embedding model's vectors find, and how a
record's vector follows its text through every write."""

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from pybodn import Collection, Embedder, MemoryEngine, NewMemory, SearchHit
from pybodn._storage.base import Store
from pybodn.errors import ConfigurationError, EmbeddingError, RecordNotFoundError
from tests.contract.conftest import STORES, TickingClock
from tests.contract.test_search import LOOKALIKE_NAMESPACES
from tests.fake_model import FakeModel, vector_of

pytest.importorskip("numpy")  # search by meaning computes with NumPy

MODEL = "fake-1"


@pytest.fixture
def model() -> FakeModel:
    return FakeModel()


@pytest.fixture(params=list(STORES.values()), ids=list(STORES))
def store(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[Store]:
    make_store: Callable[[Path], Store] = request.param
    store = make_store(tmp_path)
    yield store
    store.close()


def engine_on(store: Store, embedder: Embedder | None = None) -> MemoryEngine:
    """An engine on ``store``, as each process opens the same file. The store outlives it."""
    return MemoryEngine(store, clock=TickingClock(), embedder=embedder)


@pytest.fixture
def engine(store: Store, model: FakeModel) -> MemoryEngine:
    return engine_on(store, Embedder(MODEL, model))


@pytest.fixture
def plain(store: Store) -> Collection:
    """The same records through an engine without a model."""
    return engine_on(store).collection("user:123")


def by_meaning(notes: Collection, query: str, **options: Any) -> list[SearchHit]:
    return notes.search(query, mode="vector", **options)


def found(notes: Collection, query: str, **options: Any) -> list[str]:
    return [hit.record.content for hit in by_meaning(notes, query, **options)]


def broken(texts: list[str]) -> list[list[float]]:
    raise RuntimeError("the model is down")


# what a search finds ---------------------------------------------------------------------------


def test_the_records_closest_in_meaning_come_first(notes: Collection) -> None:
    # Arrange
    for text in ["чай", "кава кава", "кава"]:
        notes.add(text)

    # Act
    hits = by_meaning(notes, "кава")

    # Assert
    assert [hit.record.content for hit in hits] == ["кава", "кава кава", "чай"]
    assert hits[0].score == pytest.approx(1.0)
    assert hits[0].score > hits[1].score > hits[2].score


def test_a_hit_holds_the_record_as_stored(notes: Collection) -> None:
    # Arrange
    record = notes.add("кава", tags=["drink"], metadata={"hot": True})

    # Act
    [hit] = by_meaning(notes, "чай")

    # Assert
    assert hit.record == record


def test_the_score_is_the_similarity_from_minus_one_to_one(store: Store) -> None:
    # Arrange: a model that knows only three directions
    directions = {"north": [0.0, 2.0], "east": [3.0, 0.0], "south": [0.0, -1.0]}
    engine = engine_on(store, Embedder("compass", lambda texts: [directions[t] for t in texts]))
    notes = engine.collection("user:123")
    for text in directions:
        notes.add(text)

    # Act
    hits = by_meaning(notes, "north")

    # Assert
    assert [(hit.record.content, hit.score) for hit in hits] == [
        ("north", 1.0),
        ("east", 0.0),
        ("south", -1.0),
    ]


def test_every_record_is_found_however_far_without_a_threshold(notes: Collection) -> None:
    # Arrange
    notes.add("арахіс")
    notes.add("python")

    # Act / Assert
    assert len(by_meaning(notes, "кава")) == 2


def test_records_equally_close_keep_the_order_added(notes: Collection) -> None:
    # Arrange
    records = [notes.add("кава", metadata={"n": number}) for number in range(4)]
    notes.update(records[0].id, tags=["changed"])  # a change keeps its place

    # Act
    hits = by_meaning(notes, "кава")

    # Assert
    assert [hit.record.metadata["n"] for hit in hits] == [0, 1, 2, 3]


def test_limit_keeps_the_closest(notes: Collection) -> None:
    # Arrange
    for text in ["чай", "кава", "молоко", "кава кава", "арахіс"]:
        notes.add(text)

    # Act / Assert
    assert found(notes, "кава", limit=2) == ["кава", "кава кава"]


def test_a_search_returns_ten_records_unless_told_otherwise(notes: Collection) -> None:
    # Arrange
    notes.add_many([NewMemory(f"кава {number}") for number in range(12)])

    # Act / Assert
    assert len(by_meaning(notes, "кава")) == 10


def test_an_empty_collection_finds_nothing(notes: Collection) -> None:
    assert by_meaning(notes, "кава") == []


def test_records_of_other_namespaces_are_never_found(engine: MemoryEngine) -> None:
    # Arrange
    for namespace in LOOKALIKE_NAMESPACES:
        engine.collection(namespace).add("кава")

    for namespace in LOOKALIKE_NAMESPACES:
        # Act
        hits = by_meaning(engine.collection(namespace), "кава")

        # Assert
        assert [hit.record.namespace for hit in hits] == [namespace]


def test_filters_keep_the_records_that_list_keeps(notes: Collection) -> None:
    # Arrange: the same text in every record, so that all score alike and keep the order added
    records = [
        notes.add("кава", tags=["drink"], metadata={"hot": True}, importance=0.9),
        notes.add("кава", tags=["drink"], metadata={"hot": False}, importance=0.1),
        notes.add("кава", metadata={"hot": True}, importance=0.5),
        notes.add("кава", tags=["drink", "cold"], importance=0.7),
    ]
    every_filter: list[dict[str, Any]] = [
        {"tags": ["drink"]},
        {"tags": ["drink", "cold"]},
        {"metadata": {"hot": True}},
        {"min_importance": 0.5},
        {"created_since": records[1].created_at},
        {"created_before": records[2].created_at},
        {
            "tags": ["drink"],
            "metadata": {"hot": True},
            "min_importance": 0.5,
            "created_since": records[0].created_at,
            "created_before": records[3].created_at,
        },
    ]

    for filters in every_filter:
        # Act
        hits = by_meaning(notes, "кава", limit=len(records), **filters)

        # Assert
        kept = notes.list(**filters)
        assert 0 < len(kept) < len(records), filters
        assert [hit.record for hit in hits] == kept, filters


@pytest.mark.parametrize("query", ["", "   ", "?!", "🙂"])
def test_a_query_without_words_finds_nothing_and_never_calls_the_model(
    notes: Collection, model: FakeModel, query: str
) -> None:
    # Arrange
    notes.add("кава")
    model.calls.clear()

    # Act / Assert
    assert by_meaning(notes, query) == []
    assert model.calls == []


def test_a_long_query_is_taken_whole(notes: Collection, model: FakeModel) -> None:
    # Arrange: more different words than a search by words takes
    notes.add("кава")
    query = " ".join(f"w{number}" for number in range(150)) + " кава"
    model.calls.clear()

    # Act
    hits = by_meaning(notes, query)

    # Assert
    assert [hit.record.content for hit in hits] == ["кава"]
    assert model.calls == [[query]]


def test_a_search_by_meaning_needs_a_model(plain: Collection) -> None:
    with pytest.raises(ConfigurationError, match="needs an embedding model"):
        by_meaning(plain, "кава")


def test_text_and_auto_modes_still_search_by_words(notes: Collection, model: FakeModel) -> None:
    # Arrange
    notes.add("кава")
    notes.add("чай")
    model.calls.clear()

    # Act / Assert
    for mode in ("text", "auto"):
        assert [hit.record.content for hit in notes.search("кава", mode=mode)] == ["кава"]
    assert model.calls == []


# how a record's vector follows its text --------------------------------------------------------


def test_a_record_added_without_a_model_is_not_found_by_meaning(
    notes: Collection, plain: Collection
) -> None:
    # Arrange
    plain.add("кава")

    # Act / Assert
    assert by_meaning(notes, "кава") == []


def test_a_deleted_record_is_no_longer_found(notes: Collection) -> None:
    # Arrange
    record = notes.add("кава")

    # Act
    notes.delete(record.id)

    # Assert
    assert by_meaning(notes, "кава") == []


@pytest.mark.parametrize("write", ["update", "put"])
def test_a_new_text_gets_a_new_vector(notes: Collection, write: str) -> None:
    # Arrange
    record = notes.put("drink", "кава")

    # Act
    if write == "update":
        notes.update(record.id, content="чай")
    else:
        notes.put("drink", "чай")

    # Assert
    [hit] = by_meaning(notes, "чай")
    assert hit.record.content == "чай"
    assert hit.score == pytest.approx(1.0)


@pytest.mark.parametrize("write", ["update", "put"])
def test_a_text_changed_without_the_model_loses_its_vector(
    notes: Collection, plain: Collection, write: str
) -> None:
    # Arrange
    record = notes.put("drink", "кава")

    # Act
    if write == "update":
        plain.update(record.id, content="чай")
    else:
        plain.put("drink", "чай")

    # Assert
    assert by_meaning(notes, "чай") == []


def test_a_change_without_the_model_that_keeps_the_text_keeps_the_vector(
    notes: Collection, plain: Collection
) -> None:
    # Arrange
    record = notes.add("кава")

    # Act
    plain.update(record.id, tags=["drink"], importance=0.9)

    # Assert
    assert found(notes, "кава") == ["кава"]


WRITES_OF_THE_SAME_TEXT: dict[str, Callable[[Collection, str], object]] = {
    "update-of-tags": lambda notes, record_id: notes.update(record_id, tags=["drink"]),
    "update-with-the-same-text": lambda notes, record_id: notes.update(record_id, content="кава"),
    "put-of-the-same-text": lambda notes, record_id: notes.put("drink", "кава", tags=["drink"]),
}


@pytest.mark.parametrize(
    "write", WRITES_OF_THE_SAME_TEXT.values(), ids=WRITES_OF_THE_SAME_TEXT.keys()
)
def test_a_write_that_keeps_the_text_never_calls_the_model_and_keeps_the_vector(
    notes: Collection, model: FakeModel, write: Callable[[Collection, str], object]
) -> None:
    # Arrange
    record = notes.put("drink", "кава")
    model.calls.clear()

    # Act
    write(notes, record.id)

    # Assert
    assert model.calls == []
    assert found(notes, "кава") == ["кава"]


def test_each_write_of_a_new_text_calls_the_model_once(notes: Collection, model: FakeModel) -> None:
    # Act
    record = notes.add("кава")
    notes.put("drink", "чай")
    notes.put("drink", "молоко")
    notes.update(record.id, content="арахіс")

    # Assert
    assert model.calls == [["кава"], ["чай"], ["молоко"], ["арахіс"]]


def test_an_update_of_a_missing_record_never_calls_the_model(
    notes: Collection, model: FakeModel
) -> None:
    with pytest.raises(RecordNotFoundError):
        notes.update("0" * 32, content="кава")
    assert model.calls == []


def test_add_many_gives_the_model_batch_size_texts_at_a_time(
    store: Store, model: FakeModel
) -> None:
    # Arrange
    notes = engine_on(store, Embedder(MODEL, model, batch_size=2)).collection("user:123")

    # Act
    notes.add_many([NewMemory(text) for text in ["кава", "чай", "молоко", "арахіс", "python"]])

    # Assert
    assert model.calls == [["кава", "чай"], ["молоко", "арахіс"], ["python"]]
    assert found(notes, "молоко", limit=1) == ["молоко"]


def test_an_empty_add_many_never_calls_the_model(notes: Collection, model: FakeModel) -> None:
    assert notes.add_many([]) == []
    assert model.calls == []


FAILING_WRITES: dict[str, Callable[[Collection, str], object]] = {
    "add": lambda notes, record_id: notes.add("кава"),
    "add_many": lambda notes, record_id: notes.add_many([NewMemory("кава"), NewMemory("чай")]),
    "put-new": lambda notes, record_id: notes.put("drink", "кава"),
    "put-existing": lambda notes, record_id: notes.put("food", "кава"),
    "update": lambda notes, record_id: notes.update(record_id, content="кава"),
}


@pytest.mark.parametrize("write", FAILING_WRITES.values(), ids=FAILING_WRITES.keys())
@pytest.mark.parametrize(
    ("embed", "error"),
    [
        pytest.param(broken, RuntimeError, id="failing-model"),
        pytest.param(lambda texts: [], EmbeddingError, id="unusable-answer"),
    ],
)
def test_a_write_whose_model_fails_writes_nothing(
    store: Store,
    plain: Collection,
    write: Callable[[Collection, str], object],
    embed: Callable[[list[str]], list[list[float]]],
    error: type[Exception],
) -> None:
    # Arrange
    existing = plain.put("food", "арахіс")
    notes = engine_on(store, Embedder(MODEL, embed)).collection("user:123")
    before = plain.list()

    # Act / Assert
    with pytest.raises(error):
        write(notes, existing.id)
    assert plain.list() == before


def test_a_model_of_another_size_under_the_same_name_is_refused(
    store: Store, notes: Collection
) -> None:
    # Arrange
    notes.add("кава")
    resized = engine_on(store, Embedder(MODEL, lambda texts: [[1.0, 2.0] for _ in texts]))
    others = resized.collection("user:123")
    message = f"gave a vector of length 2, but its stored vectors have length {len(vector_of('a'))}"

    # Act / Assert
    with pytest.raises(EmbeddingError, match=message):
        others.add("чай")
    with pytest.raises(EmbeddingError, match=message):
        by_meaning(others, "кава")
    assert [record.content for record in notes.list()] == ["кава"]


def test_a_model_without_stored_vectors_finds_nothing(store: Store, notes: Collection) -> None:
    # Arrange
    notes.add("кава")
    other = engine_on(store, Embedder("fake-2", FakeModel())).collection("user:123")

    # Act / Assert
    assert by_meaning(other, "кава") == []


def test_once_every_vector_of_a_model_is_deleted_its_size_may_change(
    store: Store, notes: Collection, plain: Collection
) -> None:
    # Arrange: a record without a vector keeps the namespace in use
    plain.add("арахіс")
    notes.delete(notes.add("кава").id)
    resized = engine_on(store, Embedder(MODEL, lambda texts: [[1.0, 2.0] for _ in texts]))
    others = resized.collection("user:123")

    # Act
    others.add("чай")

    # Assert
    assert found(others, "чай") == ["чай"]


# embed_missing ---------------------------------------------------------------------------------


def test_embed_missing_gives_records_their_vectors_and_says_how_many(
    engine: MemoryEngine, plain: Collection
) -> None:
    # Arrange
    plain.add("кава")
    plain.add("чай")
    notes = engine.collection("user:123")
    notes.add("молоко")

    # Act
    written = engine.embed_missing()

    # Assert
    assert written == 2
    assert found(notes, "кава", limit=1) == ["кава"]
    assert sorted(found(notes, "кава")) == ["кава", "молоко", "чай"]


def test_embed_missing_goes_through_every_namespace_in_the_order_added(
    store: Store, model: FakeModel
) -> None:
    # Arrange
    without_model = engine_on(store)
    for namespace, text in [("a", "кава"), ("b", "чай"), ("a", "молоко")]:
        without_model.collection(namespace).add(text)
    engine = engine_on(store, Embedder(MODEL, model, batch_size=2))

    # Act
    written = engine.embed_missing()

    # Assert
    assert written == 3
    assert model.calls == [["кава", "чай"], ["молоко"]]
    assert found(engine.collection("b"), "чай") == ["чай"]


def test_embed_missing_with_nothing_missing_never_calls_the_model(
    engine: MemoryEngine, model: FakeModel
) -> None:
    # Arrange
    engine.collection("user:123").add("кава")
    model.calls.clear()

    # Act / Assert
    assert engine.embed_missing() == 0
    assert model.calls == []


def test_embed_missing_replaces_the_vectors_of_another_model(
    store: Store, notes: Collection
) -> None:
    # Arrange
    notes.add("кава")
    notes.add("чай")
    newer = engine_on(store, Embedder("fake-2", FakeModel()))

    # Act
    written = newer.embed_missing()

    # Assert
    assert written == 2
    assert found(newer.collection("user:123"), "чай", limit=1) == ["чай"]
    assert by_meaning(notes, "чай") == []  # the old model's vectors are gone


@pytest.mark.parametrize("place", [0, 1], ids=["first-of-the-batch", "last-of-the-batch"])
def test_a_record_whose_text_changes_meanwhile_is_left_for_next_time(
    store: Store, plain: Collection, place: int
) -> None:
    # Arrange: another process changes the text while the model works on the old one
    records = [plain.add("кава"), plain.add("чай")]
    changed, unchanged = records[place], records[1 - place]
    model = FakeModel()

    def embed(texts: list[str]) -> list[list[float]]:
        if changed.content in texts:
            plain.update(changed.id, content="молоко")
        return model(texts)

    engine = engine_on(store, Embedder(MODEL, embed))
    notes = engine.collection("user:123")

    # Act / Assert
    assert engine.embed_missing() == 1
    assert found(notes, "молоко") == [unchanged.content]  # "молоко" has no vector yet
    assert engine.embed_missing() == 1
    assert found(notes, "молоко", limit=1) == ["молоко"]


def test_a_record_deleted_meanwhile_is_skipped(store: Store, plain: Collection) -> None:
    # Arrange
    first = plain.add("кава")
    plain.add("чай")
    model = FakeModel()

    def embed(texts: list[str]) -> list[list[float]]:
        plain.delete(first.id)
        return model(texts)

    engine = engine_on(store, Embedder(MODEL, embed))

    # Act / Assert
    assert engine.embed_missing() == 1
    assert found(engine.collection("user:123"), "кава") == ["чай"]


def test_records_added_while_it_runs_are_left_for_next_time(
    store: Store, plain: Collection
) -> None:
    # Arrange
    plain.add("кава")
    model = FakeModel()

    def embed(texts: list[str]) -> list[list[float]]:
        if texts == ["кава"]:
            plain.add("чай")
        return model(texts)

    engine = engine_on(store, Embedder(MODEL, embed, batch_size=1))

    # Act / Assert
    assert engine.embed_missing() == 1
    assert model.calls == [["кава"]]
    assert engine.embed_missing() == 1
    assert model.calls == [["кава"], ["чай"]]


def test_a_failing_model_keeps_the_batches_already_written(store: Store, plain: Collection) -> None:
    # Arrange: the model fails on its second batch
    plain.add("кава")
    plain.add("чай")
    model = FakeModel()

    def embed(texts: list[str]) -> list[list[float]]:
        if texts == ["чай"]:
            raise RuntimeError("the model is down")
        return model(texts)

    engine = engine_on(store, Embedder(MODEL, embed, batch_size=1))

    # Act / Assert
    with pytest.raises(RuntimeError, match="the model is down"):
        engine.embed_missing()
    assert found(engine.collection("user:123"), "кава") == ["кава"]


def test_embed_missing_needs_a_model(store: Store) -> None:
    with pytest.raises(ConfigurationError, match="needs an embedding model"):
        engine_on(store).embed_missing()
