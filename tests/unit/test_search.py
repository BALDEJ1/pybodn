"""The terms a query looks for, and how records score against them, before any store."""

import pytest

from pybodn._search import MAX_QUERY_WORDS, Term, bm25_scores, query_terms
from pybodn.errors import ValidationError


def exact(text: str) -> Term:
    return Term(text, is_prefix=False)


def prefix(text: str) -> Term:
    return Term(text, is_prefix=True)


# query_terms -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "terms"),
    [
        pytest.param("алергії", (exact("алергії"), prefix("алергі")), id="known-ending"),
        pytest.param("лікар", (exact("лікар"), prefix("лікар")), id="long-without-ending"),
        pytest.param("сайт", (exact("сайт"), prefix("сайт")), id="four-letters-without-ending"),
        pytest.param("про", (exact("про"),), id="short-without-ending"),
        pytest.param("має", (exact("має"),), id="ending-would-leave-two-letters"),
        pytest.param("10000", (exact("10000"),), id="number"),
        pytest.param("2024р", (exact("2024р"), prefix("2024р")), id="number-with-a-letter"),
        pytest.param("займається", (exact("займається"), prefix("займа")), id="reflexive"),
        pytest.param("вчився", (exact("вчився"), prefix("вчи")), id="reflexive-past"),
        pytest.param("downloads", (exact("downloads"), prefix("download")), id="english"),
        pytest.param("हिन्दी", (exact("हिन्दी"), prefix("हिन्दी")), id="marks-stay-in-the-word"),
    ],
)
def test_a_word_looks_for_itself_and_the_start_its_forms_share(
    query: str, terms: tuple[Term, ...]
) -> None:
    assert query_terms(query) == terms


def test_words_are_prepared_as_stored_words_are() -> None:
    assert query_terms("П\u2019Є АЛЕРГІ\u0301Ї") == query_terms("пє алергії")


def test_each_term_comes_once_in_the_order_of_the_query() -> None:
    # Act: "алергії" and "алергію" share a start; "кава" comes twice
    terms = query_terms("кава алергії, алергію і кава")

    # Assert
    assert terms == (
        exact("кава"),
        prefix("кав"),
        exact("алергії"),
        prefix("алергі"),
        exact("алергію"),
        exact("і"),
    )


@pytest.mark.parametrize("query", ["", "   ", "?!", "—"])
def test_a_query_without_words_has_no_terms(query: str) -> None:
    assert query_terms(query) == ()


def test_a_query_may_hold_100_different_words_however_often_each_is_repeated() -> None:
    # Arrange
    hundred = [f"w{number}" for number in range(MAX_QUERY_WORDS)]

    # Act
    terms = query_terms(" ".join(hundred * 3))

    # Assert
    assert [term.text for term in terms if not term.is_prefix] == hundred


def test_a_query_of_more_than_100_different_words_is_rejected() -> None:
    # Arrange
    query = " ".join(f"w{number}" for number in range(MAX_QUERY_WORDS + 1))

    # Act / Assert
    with pytest.raises(ValidationError, match="^query must not hold more than 100 different words"):
        query_terms(query)


# Term ------------------------------------------------------------------------------------------


def test_an_exact_term_counts_only_the_same_word() -> None:
    assert exact("кава").count_in(["кава", "каву", "кава", "кав"]) == 2


def test_a_prefix_term_counts_every_word_that_starts_with_it() -> None:
    assert prefix("кав").count_in(["кава", "каву", "кав", "ка", "какао"]) == 3


# bm25_scores -----------------------------------------------------------------------------------


def test_a_store_without_records_has_no_scores() -> None:
    assert bm25_scores([exact("кава")], []) == []


def test_a_record_no_term_finds_has_no_score() -> None:
    # Act
    scores = bm25_scores([exact("кава")], [["кава"], ["чай"], []])

    # Assert
    assert scores[0] is not None
    assert scores[1:] == [None, None]


def test_more_of_a_word_scores_higher() -> None:
    # Act
    once, twice, _, _ = bm25_scores(
        [exact("кава")], [["кава", "чай"], ["кава", "кава"], ["сік"], ["вода"]]
    )

    # Assert
    assert once is not None
    assert twice is not None
    assert twice > once


def test_a_longer_record_with_the_same_word_scores_lower() -> None:
    # Act
    short, long, _, _ = bm25_scores(
        [exact("кава")], [["кава"], ["кава", "з", "молоком", "і", "цукром"], ["сік"], ["вода"]]
    )

    # Assert
    assert short is not None
    assert long is not None
    assert short > long


def test_a_rarer_word_scores_higher() -> None:
    # Act
    common, rare, _, _ = bm25_scores(
        [exact("кава"), exact("арахіс")], [["кава"], ["арахіс"], ["кава"], ["сік"]]
    )

    # Assert
    assert common is not None
    assert rare is not None
    assert rare > common


def test_a_word_in_more_than_half_of_the_records_still_scores_above_zero() -> None:
    # Act
    scores = bm25_scores([exact("кава")], [["кава"], ["кава"], ["кава", "чай"]])

    # Assert
    assert all(score is not None and 0 < score < 1e-5 for score in scores)
