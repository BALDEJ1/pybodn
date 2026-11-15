"""search() finds the records that hold words of the query, best first."""

from typing import Any

import pytest

from pybodn import Collection, MemoryEngine, SearchHit
from pybodn._words import words
from pybodn.errors import ConfigurationError, ValidationError

DEFAULT_LIMIT = 10
MAX_QUERY_WORDS = 100


def found(notes: Collection, query: str, **options: Any) -> list[str]:
    """The content of each record the search finds, best first."""
    return [hit.record.content for hit in notes.search(query, **options)]


# what is found -------------------------------------------------------------------------------


def test_a_search_finds_every_record_that_holds_a_word_of_the_query(notes: Collection) -> None:
    # Arrange
    notes.add("Алергія на арахіс")
    notes.add("П’є каву без цукру")
    notes.add("Працює лікарем")

    # Act / Assert
    assert sorted(found(notes, "арахіс каву")) == ["Алергія на арахіс", "П’є каву без цукру"]


def test_a_hit_holds_the_record_as_stored_and_a_score_above_zero(notes: Collection) -> None:
    # Arrange
    record = notes.add("Алергія на арахіс", tags=["health"], metadata={"since": 2020})

    # Act
    (hit,) = notes.search("арахіс")

    # Assert
    assert isinstance(hit, SearchHit)
    assert hit.record == record
    assert hit.score > 0


def test_case_stress_marks_and_apostrophes_do_not_matter(notes: Collection) -> None:
    # Arrange
    notes.add("Алергія на арахіс")
    notes.add("П’є каву")

    # Act / Assert
    assert found(notes, "АЛЕРГІ\u0301Я") == ["Алергія на арахіс"]
    assert found(notes, "п'є") == ["П’є каву"]


@pytest.mark.parametrize(
    ("query", "stored"),
    [
        pytest.param("алергії", "Алергія на арахіс", id="known-ending"),
        pytest.param("лікар", "Записався до лікаря", id="long-word-without-ending"),
        pytest.param("займається", "Любить займатися йогою", id="reflexive-verb"),
        pytest.param("projects", "Works on a new project", id="english"),
    ],
)
def test_a_word_finds_its_other_forms(notes: Collection, query: str, stored: str) -> None:
    # Arrange
    notes.add(stored)
    notes.add("Інший запис")

    # Act / Assert
    assert found(notes, query) == [stored]


def test_a_short_word_finds_only_itself(notes: Collection) -> None:
    # Arrange
    notes.add("Новий проєкт")
    notes.add("Розповідь про відпустку")

    # Act / Assert
    assert found(notes, "про") == ["Розповідь про відпустку"]


@pytest.mark.parametrize("query", ["", "   ", "?!", "—"])
def test_a_query_without_words_finds_nothing(notes: Collection, query: str) -> None:
    # Arrange
    notes.add("Кава")

    # Act / Assert
    assert notes.search(query) == []


def test_a_word_that_no_record_holds_finds_nothing(notes: Collection) -> None:
    # Arrange
    notes.add("Кава з молоком")

    # Act / Assert
    assert notes.search("чай") == []


def test_an_empty_collection_finds_nothing(notes: Collection) -> None:
    assert notes.search("кава") == []


# Namespaces that differ by little: a prefix, case, a space, quotes, a NUL character, letters
# beyond ASCII, and two longer than 16 KB that differ only at the end.
LOOKALIKE_NAMESPACES = [
    "user:1",
    "user:10",
    "User:1",
    "user:1 ",
    'user:"1"',
    "user:1\x00",
    "користувач:1",
    "a" * 20_000,
    "a" * 19_999 + "b",
]


def test_records_of_other_namespaces_are_never_found(engine: MemoryEngine) -> None:
    # Arrange
    for namespace in LOOKALIKE_NAMESPACES:
        engine.collection(namespace).add("Кава з молоком")

    for namespace in LOOKALIKE_NAMESPACES:
        # Act
        hits = engine.collection(namespace).search("кава")

        # Assert
        assert [hit.record.namespace for hit in hits] == [namespace]


@pytest.mark.parametrize(
    "query",
    [
        '"кава" OR чай*',
        "NEAR(кава чай)",
        "кава AND NOT чай",
        "words:кава",
        "{words}: кава",
        "^кава",
        "(кава",
        '"',
        "*",
        "кава' OR '1'='1",
    ],
)
def test_search_syntax_in_a_query_is_only_words(notes: Collection, query: str) -> None:
    # Arrange
    for text in ["Кава", "Чай", "Чайник", "Not", "Or else", "Words", "Near"]:
        notes.add(text)

    # Act / Assert
    assert found(notes, query) == found(notes, " ".join(words(query)))


def test_a_deleted_record_is_no_longer_found(notes: Collection) -> None:
    # Arrange
    record = notes.add("Кава з молоком")

    # Act
    notes.delete(record.id)

    # Assert
    assert notes.search("кава") == []


def test_a_record_is_found_by_its_new_text_only(notes: Collection) -> None:
    # Arrange
    record = notes.add("Кава з молоком")

    # Act
    notes.update(record.id, content="Чай з медом")

    # Assert
    assert notes.search("кава") == []
    assert found(notes, "чай") == ["Чай з медом"]


# order ---------------------------------------------------------------------------------------


def test_the_exact_form_ranks_above_another_form(notes: Collection) -> None:
    # Arrange: "кавун" also starts with "кав", the start that "каву" shares with its forms
    notes.add("Купив кавун")
    notes.add("Купив каву")

    # Act / Assert
    assert found(notes, "каву") == ["Купив каву", "Купив кавун"]


def test_more_words_of_the_query_rank_higher(notes: Collection) -> None:
    # Arrange: records of one length, so that only the words they share with the query differ
    notes.add("кава чай сік")
    notes.add("кава молоко цукор")
    notes.add("вода чай сік")
    notes.add("кава молоко сік")

    # Act / Assert
    assert found(notes, "кава молоко цукор") == [
        "кава молоко цукор",
        "кава молоко сік",
        "кава чай сік",
    ]


def test_a_rarer_word_ranks_higher(notes: Collection) -> None:
    # Arrange
    notes.add("Кава зранку")
    notes.add("Кава вдень")
    notes.add("Арахіс зранку")
    notes.add("Кава ввечері")

    # Act / Assert
    assert found(notes, "кава арахіс")[0] == "Арахіс зранку"


def test_a_word_in_more_than_half_of_the_records_still_ranks_them_rightly(
    notes: Collection,
) -> None:
    # Arrange: "кава" is in every record, so it weighs almost nothing
    notes.add("Кава без цукру")
    notes.add("Кава з молоком")
    notes.add("Кава і чай")

    # Act / Assert
    assert found(notes, "кава молоко") == ["Кава з молоком", "Кава без цукру", "Кава і чай"]


def test_equal_scores_keep_the_order_added(notes: Collection) -> None:
    # Arrange
    records = [notes.add("Кава", metadata={"n": number}) for number in range(5)]
    notes.update(records[0].id, tags=["changed"])  # a change keeps its place

    # Act
    hits = notes.search("кава")

    # Assert
    assert [hit.record.metadata["n"] for hit in hits] == [0, 1, 2, 3, 4]


def test_limit_keeps_the_best_records(notes: Collection) -> None:
    # Arrange
    notes.add("кава чай сік")
    notes.add("кава молоко цукор")
    notes.add("кава молоко сік")

    # Act / Assert
    assert found(notes, "кава молоко цукор", limit=2) == ["кава молоко цукор", "кава молоко сік"]


def test_a_search_returns_ten_records_unless_told_otherwise(notes: Collection) -> None:
    # Arrange
    for number in range(DEFAULT_LIMIT + 2):
        notes.add(f"Кава {number}")

    # Act / Assert
    assert len(notes.search("кава")) == DEFAULT_LIMIT
    assert len(notes.search("кава", limit=DEFAULT_LIMIT + 5)) == DEFAULT_LIMIT + 2


# filters and modes ---------------------------------------------------------------------------


def test_filters_keep_the_records_that_list_keeps(notes: Collection) -> None:
    # Arrange: the same text in every record, so that all score alike and keep the order added
    records = [
        notes.add("Кава", tags=["drink"], metadata={"hot": True}, importance=0.9),
        notes.add("Кава", tags=["drink"], metadata={"hot": False}, importance=0.1),
        notes.add("Кава", metadata={"hot": True}, importance=0.5),
        notes.add("Кава", tags=["drink", "cold"], importance=0.7),
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
        hits = notes.search("кава", limit=len(records), **filters)

        # Assert
        kept = notes.list(**filters)
        assert 0 < len(kept) < len(records), filters
        assert [hit.record for hit in hits] == kept, filters
    assert len(notes.search("кава", tags=None, min_importance=None)) == len(records)


def test_text_and_auto_modes_search_by_words(notes: Collection) -> None:
    # Arrange
    notes.add("Кава з молоком")
    notes.add("Чай")

    # Act / Assert
    assert notes.search("кава", mode="text") == notes.search("кава", mode="auto")
    assert notes.search("кава", mode="auto") == notes.search("кава")


@pytest.mark.parametrize("query", ["кава", ""])
def test_vector_mode_needs_an_embedding_model(notes: Collection, query: str) -> None:
    # Arrange
    notes.add("Кава")

    # Act / Assert
    with pytest.raises(ConfigurationError, match="embedding model"):
        notes.search(query, mode="vector")


def test_a_query_of_more_than_100_different_words_is_rejected(notes: Collection) -> None:
    # Arrange
    query = " ".join(f"w{number}" for number in range(MAX_QUERY_WORDS + 1))

    # Act / Assert
    with pytest.raises(ValidationError, match="^query must not hold more than 100"):
        notes.search(query)
