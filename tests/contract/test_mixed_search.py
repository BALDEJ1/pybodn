"""Mixed search, the same in every store: what ``mode="auto"`` finds with an embedding model, and
how it ranks the records that the search by words and the search by meaning found as one."""

from collections.abc import Callable

import pytest

from pybodn import Collection, Embedder, MemoryEngine, NewMemory, SearchHit
from pybodn._storage.base import Store
from pybodn.errors import EmbeddingError, ValidationError
from tests.contract.conftest import MODEL, engine_on
from tests.contract.test_search import LOOKALIKE_NAMESPACES
from tests.fake_model import FakeModel, broken

pytest.importorskip("numpy")  # search by meaning computes with NumPy


@pytest.fixture
def embedder(model: FakeModel) -> Embedder:
    return Embedder(MODEL, model)


def mapped(store: Store, vectors: dict[str, list[float]]) -> Collection:
    """The records of ``store`` through an engine whose model gives each text the vector that
    ``vectors`` holds for it, so that a test sets each record's place by meaning."""
    model = Embedder("mapped", lambda texts: [vectors[text] for text in texts])
    return engine_on(store, model).collection("user:123")


def contents(hits: list[SearchHit]) -> list[str]:
    return [hit.record.content for hit in hits]


# what a search finds ---------------------------------------------------------------------------


def test_a_record_is_found_by_meaning_without_a_word_of_the_query(
    notes: Collection, model: FakeModel
) -> None:
    # Arrange
    notes.add("молоко")
    model.calls.clear()

    # Act
    hits = notes.search("кава")

    # Assert: first, and found one way only
    assert contents(hits) == ["молоко"]
    assert hits[0].score == 1 / 61
    assert model.calls == [["кава"]]


def test_a_record_without_a_vector_of_the_model_is_still_found_by_words(
    notes: Collection, plain: Collection
) -> None:
    # Arrange: written without the model, and not yet given a vector by embed_missing
    plain.add("кава")
    plain.add("чай")

    # Act
    hits = notes.search("кава")

    # Assert
    assert contents(hits) == ["кава"]
    assert hits[0].score == 1 / 61


def test_a_hit_holds_the_record_as_stored(notes: Collection) -> None:
    # Arrange
    record = notes.add("кава", tags=["drink"], metadata={"hot": True})

    # Act
    [hit] = notes.search("кава")

    # Assert
    assert hit.record == record


def test_an_empty_collection_finds_nothing(notes: Collection) -> None:
    assert notes.search("кава") == []


def test_records_of_other_namespaces_are_never_found(engine: MemoryEngine) -> None:
    # Arrange
    for namespace in LOOKALIKE_NAMESPACES:
        engine.collection(namespace).add("кава")

    for namespace in LOOKALIKE_NAMESPACES:
        # Act
        hits = engine.collection(namespace).search("кава")

        # Assert
        assert [hit.record.namespace for hit in hits] == [namespace]


def test_filters_keep_only_the_records_that_list_keeps(notes: Collection) -> None:
    # Arrange: without the filter, the search by words would also find the second record, and
    # the search by meaning the second and the third
    kept = notes.add("кава", tags=["drink"])
    notes.add("кава")
    notes.add("молоко")

    # Act / Assert
    assert [hit.record for hit in notes.search("кава", tags=["drink"])] == [kept]


@pytest.mark.parametrize("query", ["", "   ", "?!", "🙂"])
def test_a_query_without_words_finds_nothing_and_never_calls_the_model(
    notes: Collection, model: FakeModel, query: str
) -> None:
    # Arrange
    notes.add("кава")
    model.calls.clear()

    # Act / Assert
    assert notes.search(query) == []
    assert model.calls == []


def test_a_query_of_more_than_100_different_words_is_refused_before_the_model_is_called(
    notes: Collection, model: FakeModel
) -> None:
    # Arrange
    notes.add("кава")
    query = " ".join(f"w{number}" for number in range(101))
    model.calls.clear()

    # Act / Assert
    with pytest.raises(ValidationError, match="more than 100 different words"):
        notes.search(query)
    assert model.calls == []


@pytest.mark.parametrize(
    ("embed", "error", "message"),
    [
        pytest.param(broken, RuntimeError, "the model is down", id="failing-model"),
        pytest.param(lambda texts: [], EmbeddingError, "one vector per text", id="unusable-answer"),
    ],
)
def test_a_search_whose_model_fails_raises_its_error(
    store: Store,
    plain: Collection,
    embed: Callable[[list[str]], list[list[float]]],
    error: type[Exception],
    message: str,
) -> None:
    # Arrange: a record the search by words alone would find
    plain.add("кава")
    notes = engine_on(store, Embedder(MODEL, embed)).collection("user:123")

    # Act / Assert
    with pytest.raises(error, match=message):
        notes.search("кава")


# how the records found rank ----------------------------------------------------------------------


def test_a_record_found_both_ways_rises_above_the_first_of_each(
    store: Store, plain: Collection
) -> None:
    # Arrange: by words, "кава" comes first and "кава з молоком" second; by meaning, "латте"
    # first and "кава з молоком" second. "кава" has no vector, and "латте" no word of the query.
    plain.add("кава")
    notes = mapped(store, {"кава": [1.0, 0.0], "латте": [1.0, 0.0], "кава з молоком": [0.8, 0.6]})
    notes.add("латте")
    notes.add("кава з молоком")

    # Act
    hits = notes.search("кава")

    # Assert: 1 / (60 + place) for each search that found a record, added up
    assert [(hit.record.content, hit.score) for hit in hits] == [
        ("кава з молоком", 1 / 62 + 1 / 62),
        ("кава", 1 / 61),
        ("латте", 1 / 61),
    ]


def test_equal_scores_keep_the_order_of_the_search_by_words(store: Store) -> None:
    # Arrange: the first record added comes first by meaning and second by words, the other the
    # other way round, so that both score 1 / 61 + 1 / 62
    notes = mapped(
        store, {"кава": [1.0, 0.0], "кава зерна свіжі": [1.0, 0.0], "кава арахіс": [0.8, 0.6]}
    )
    notes.add("кава зерна свіжі")
    notes.add("кава арахіс")

    # Act
    hits = notes.search("кава")

    # Assert
    assert contents(hits) == ["кава арахіс", "кава зерна свіжі"]
    assert hits[0].score == hits[1].score


def test_limit_keeps_the_best(notes: Collection) -> None:
    # Arrange
    for text in ["чай", "кава", "молоко", "кава кава", "арахіс", "кава чай"]:
        notes.add(text)

    # Act
    best = notes.search("кава", limit=6)

    # Assert
    assert len(best) == 6
    assert notes.search("кава", limit=2) == best[:2]


def test_a_search_returns_ten_records_unless_told_otherwise(notes: Collection) -> None:
    # Arrange
    notes.add_many([NewMemory(f"кава {number}") for number in range(12)])

    # Act / Assert
    assert len(notes.search("кава")) == 10


def test_each_search_gives_its_best_50_records_or_limit_if_more(
    store: Store, plain: Collection
) -> None:
    # Arrange: by words, 49 records without a vector come first; by meaning, 49 records without
    # the query's word. The next two, found both ways, stand 50th and 51st in each, and either
    # would outrank every record found one way.
    plain.add_many([NewMemory("кава") for _ in range(49)])
    notes = mapped(store, {"кава": [1.0, 0.0], "латте": [1.0, 0.0]})
    notes.add_many([NewMemory("латте") for _ in range(49)])
    fiftieth, fifty_first = notes.add("кава"), notes.add("кава")

    # Act
    few = notes.search("кава", limit=10)
    many = notes.search("кава", limit=51)

    # Assert
    assert few[0].record == fiftieth
    assert fifty_first not in [hit.record for hit in few]
    assert [hit.record for hit in many[:2]] == [fiftieth, fifty_first]
