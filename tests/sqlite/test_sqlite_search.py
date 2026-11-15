"""SQLite's full-text search reads only the namespace's rows, and ranks exactly as the shared
formula ranks the in-memory store."""

import random
from pathlib import Path

import pytest

from pybodn import MemoryEngine, SearchHit
from pybodn._search import Term
from pybodn._storage.sqlite import _match


def test_the_index_is_asked_for_the_namespace_and_each_term_as_quoted_text() -> None:
    # Arrange: no word holds a quote, but one would still be read as text
    terms = [Term("кава", False), Term("кав", True), Term('a"b', False)]

    # Act
    query = _match("user:1", terms)

    # Assert: "user:1" in hex
    assert query == 'namespace_token : "757365723a31" AND words : ("кава" OR "кав"* OR "a""b")'


# Forms of a few words, short and long words, and words in several languages.
VOCABULARY = (
    "кава каву кавою кавун чай чаю молоко молоком цукор лікар лікаря лікарем алергія алергії"
    " арахіс проєкт проєкту займається займатися працює працював project projects python і з"
    " на про"
).split()
NAMESPACES = ["user:1", "user:2", "user:3"]
RECORDS_PER_NAMESPACE = 40
# In two records of every three, so that it weighs the least a term can.
IN_MOST_RECORDS = "запис"
QUERIES = [
    "кава",
    "каву",
    "лікар",
    "алергії арахіс",
    "займається",
    "project",
    "про",
    "з",
    "кава молоко цукор чай",
    "чай чай",
    IN_MOST_RECORDS,
    f"{IN_MOST_RECORDS} арахіс",
]


def fill(engine: MemoryEngine) -> None:
    """The same records, in the same order, in every engine: 1 to 12 words from the vocabulary,
    after the word that is in most records in two records of every three."""
    words = random.Random(2026)  # noqa: S311 - the same test data every run, no secret
    for namespace in NAMESPACES:
        notes = engine.collection(namespace)
        for number in range(RECORDS_PER_NAMESPACE):
            text = " ".join(words.choices(VOCABULARY, k=words.randint(1, 12)))
            if number % 3:
                text = f"{IN_MOST_RECORDS} {text}"
            notes.add(text, metadata={"n": number})


def described(hits: list[SearchHit]) -> list[tuple[str, object]]:
    """Each hit as its namespace and its place among the records added there."""
    return [(hit.record.namespace, hit.record.metadata["n"]) for hit in hits]


@pytest.mark.parametrize("limit", [5, 1_000])
@pytest.mark.parametrize("query", QUERIES)
def test_sqlite_finds_and_ranks_as_the_in_memory_store_does(
    tmp_path: Path, query: str, limit: int
) -> None:
    with MemoryEngine.in_memory() as memory, MemoryEngine.sqlite(tmp_path / "m.db") as sqlite:
        # Arrange
        fill(memory)
        fill(sqlite)

        for namespace in NAMESPACES:
            # Act
            expected = memory.collection(namespace).search(query, limit=limit)
            hits = sqlite.collection(namespace).search(query, limit=limit)

            # Assert
            assert expected, "every query finds records in every namespace"
            assert described(hits) == described(expected)
            assert [hit.score for hit in hits] == [hit.score for hit in expected]


# Few words in many mixes, so that records holding different words often score alike. On a
# processor that fuses multiplication with addition, SQLite's unrounded sums for such records
# come out a last digit apart, and the records changed places.
FEW_WORDS = "лікаря cats running dogs dog кави cat".split()


@pytest.mark.parametrize("seed", range(5))
def test_records_that_score_alike_keep_the_order_added_in_both_stores(
    tmp_path: Path, seed: int
) -> None:
    with MemoryEngine.in_memory() as memory, MemoryEngine.sqlite(tmp_path / "m.db") as sqlite:
        # Arrange
        words = random.Random(seed)  # noqa: S311 - the same test data every run, no secret
        texts = [" ".join(words.choices(FEW_WORDS, k=5)) for _ in range(20)]
        for engine in (memory, sqlite):
            notes = engine.collection("user:1")
            for number, text in enumerate(texts):
                notes.add(text, metadata={"n": number})

        # Act
        expected = memory.collection("user:1").search("cats dog", limit=len(texts))
        hits = sqlite.collection("user:1").search("cats dog", limit=len(texts))

        # Assert
        assert described(hits) == described(expected)
        assert [hit.score for hit in hits] == [hit.score for hit in expected]
