"""The vectors in the file: how they are stored, that a search by meaning or a mixed one there finds
and ranks exactly as the in-memory store does, and that a vector never follows a record it was not
made for."""

import hashlib
import random
import struct
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from pybodn import Collection, Embedder, MemoryEngine, SearchHit
from pybodn._storage import schema, sqlite
from pybodn._words import WORD_RULES
from pybodn.errors import StorageError
from tests.fake_model import FakeModel
from tests.sqlite.support import run_sql

np = pytest.importorskip("numpy")

MODEL = Embedder("fake-1", FakeModel())


def found(notes: Collection, query: str) -> list[str]:
    return [hit.record.content for hit in notes.search(query, mode="vector")]


def test_a_vector_is_stored_at_unit_length_as_little_endian_floats_with_its_model(
    path: Path,
) -> None:
    # Arrange
    compass = Embedder("compass", lambda texts: [[3.0, 4.0] for _ in texts])

    # Act
    with MemoryEngine.sqlite(path, embedder=compass) as engine:
        record = engine.collection("user:123").add("north")

    # Assert
    assert run_sql(path, "SELECT seq, record_id, model, vector FROM memory_vectors") == [
        (1, record.id, "compass", struct.pack("<2f", 0.6, 0.8))
    ]


def test_the_file_holds_a_vector_only_for_each_record_that_has_one(path: Path) -> None:
    with MemoryEngine.sqlite(path, embedder=MODEL) as engine, MemoryEngine.sqlite(path) as other:
        # Arrange
        notes, plain = engine.collection("user:123"), other.collection("user:123")
        changed, deleted, tagged = notes.add("кава"), notes.add("чай"), notes.add("молоко")

        # Act
        plain.update(changed.id, content="арахіс")
        notes.delete(deleted.id)
        plain.update(tagged.id, tags=["drink"])

    # Assert
    assert run_sql(path, "SELECT record_id FROM memory_vectors") == [(tagged.id,)]


def test_records_another_process_adds_without_the_model_get_vectors_later(path: Path) -> None:
    with MemoryEngine.sqlite(path, embedder=MODEL) as engine, MemoryEngine.sqlite(path) as other:
        # Arrange
        other.collection("user:123").add("кава")

        # Act / Assert
        assert engine.embed_missing() == 1
        assert found(engine.collection("user:123"), "кава") == ["кава"]


def test_a_vector_left_by_pybodn_0_1_0_never_reaches_the_next_record_of_its_seq(
    path: Path,
) -> None:
    with MemoryEngine.sqlite(path, embedder=MODEL) as engine:
        # Arrange: 0.1.0 deletes the last record, knowing nothing of its vector, and SQLite gives
        # its seq to the next record
        notes = engine.collection("user:123")
        old = notes.add("кава")
        run_sql(path, "DELETE FROM memories WHERE id = ?", (old.id,))
        with MemoryEngine.sqlite(path) as other:
            new = other.collection("user:123").add("чай")
        assert run_sql(path, "SELECT seq FROM memory_vectors") == [(1,)]
        assert run_sql(path, "SELECT seq FROM memories") == [(1,)]

        # Act / Assert
        assert found(notes, "кава") == []
        assert engine.embed_missing() == 1
        assert found(notes, "кава") == ["чай"]
        assert run_sql(path, "SELECT record_id FROM memory_vectors") == [(new.id,)]


@pytest.mark.parametrize("damage", ["x'00'", "x'0000803f0000803f'", "'text'"])
def test_a_damaged_vector_raises_storage_error(path: Path, damage: str) -> None:
    with MemoryEngine.sqlite(path, embedder=MODEL) as engine:
        # Arrange
        notes = engine.collection("user:123")
        notes.add("кава")
        notes.add("чай")
        run_sql(path, f"UPDATE memory_vectors SET vector = {damage} WHERE seq = 2")  # noqa: S608

        # Act / Assert
        with pytest.raises(StorageError, match="damaged"):
            notes.search("кава", mode="vector")


def test_a_record_whose_content_is_no_longer_text_never_reaches_the_model(path: Path) -> None:
    with MemoryEngine.sqlite(path, embedder=MODEL) as engine:
        # Arrange: a record without a vector, whose content the file now holds as bytes
        with MemoryEngine.sqlite(path) as other:
            record = other.collection("user:123").add("кава")
        run_sql(path, "UPDATE memories SET content = x'00' WHERE id = ?", (record.id,))

        # Act / Assert
        with pytest.raises(StorageError, match=f"record '{record.id}' in the database is damaged"):
            engine.embed_missing()


def test_a_search_reads_its_namespace_in_order_and_each_vector_by_seq(path: Path) -> None:
    # Arrange
    with MemoryEngine.sqlite(path, embedder=MODEL) as engine:
        for namespace in ("user:1", "user:2"):
            engine.collection(namespace).add("кава")

    # Act
    search = f"{sqlite._SEARCH_VECTORS}{sqlite._IN_ORDER}"
    plan = run_sql(path, f"EXPLAIN QUERY PLAN {search}", ("u", "m"))

    # Assert: no sort, and no row of another namespace is read
    steps = [str(row[-1]) for row in plan]
    assert len(steps) == 2, steps
    assert "memories USING INDEX ix_memories_namespace_seq (namespace=?)" in steps[0]
    assert "memory_vectors USING INTEGER PRIMARY KEY (rowid=?)" in steps[1]


# What a newer pybodn leaves behind when it upgrades the file under a running process.
UPGRADES: dict[str, tuple[str, tuple[object, ...]]] = {
    "schema": (f"PRAGMA user_version = {len(schema.MIGRATIONS) + 1}", ()),
    "word-rules": ("UPDATE settings SET value = ? WHERE name = 'word_rules'", (WORD_RULES + 1,)),
}

VECTOR_ACTIONS: dict[str, Callable[[MemoryEngine], object]] = {
    "search": lambda engine: engine.collection("user:123").search("кава", mode="vector"),
    "add": lambda engine: engine.collection("user:123").add("чай"),
    "embed_missing": lambda engine: engine.embed_missing(),
}


@pytest.mark.parametrize("action", VECTOR_ACTIONS.values(), ids=VECTOR_ACTIONS.keys())
@pytest.mark.parametrize("upgrade", UPGRADES.values(), ids=UPGRADES.keys())
def test_a_process_overtaken_by_a_newer_pybodn_stops_before_it_touches_vectors(
    path: Path, upgrade: tuple[str, tuple[object, ...]], action: Callable[[MemoryEngine], object]
) -> None:
    with MemoryEngine.sqlite(path, embedder=MODEL) as engine:
        # Arrange
        with MemoryEngine.sqlite(path) as other:
            other.collection("user:123").add("кава")  # no vector yet, for embed_missing
        before = run_sql(path, "SELECT * FROM memory_vectors")
        run_sql(path, *upgrade)

        # Act / Assert
        with pytest.raises(StorageError, match="restart"):
            action(engine)
        assert run_sql(path, "SELECT * FROM memory_vectors") == before


# Both stores ------------------------------------------------------------------------------------

VOCABULARY = "кава чай молоко арахіс алергія python вода сік мед цукор".split()
NAMESPACES = ["user:1", "user:2", "user:3"]
RECORDS_PER_NAMESPACE = 40


def random_vectors(texts: list[str]) -> list[list[float]]:
    """A vector for each text that looks random but is always the same for the same text."""
    vectors = []
    for text in texts:
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")
        vectors.append(np.random.default_rng(seed).normal(size=16).tolist())
    return vectors


RANDOM = Embedder("random-16", random_vectors)


def fill(engine: MemoryEngine) -> None:
    """The same records, in the same order, in every engine; some of them share a text."""
    words = random.Random(2026)  # noqa: S311 - the same test data every run, no secret
    for namespace in NAMESPACES:
        notes = engine.collection(namespace)
        for number in range(RECORDS_PER_NAMESPACE):
            text = " ".join(words.choices(VOCABULARY, k=words.randint(1, 3)))
            notes.add(
                text,
                metadata={"n": number},
                tags=["even"] if number % 2 == 0 else [],
                importance=number / RECORDS_PER_NAMESPACE,
            )


def described(hits: list[SearchHit]) -> list[tuple[str, object]]:
    """Each hit as its namespace and its place among the records added there."""
    return [(hit.record.namespace, hit.record.metadata["n"]) for hit in hits]


@pytest.mark.parametrize(
    "filters", [{}, {"tags": ["even"]}, {"min_importance": 0.5}], ids=["all", "tag", "importance"]
)
@pytest.mark.parametrize("limit", [5, 1_000])
@pytest.mark.parametrize("query", ["кава", "чай молоко", "python", "зовсім інше"])
@pytest.mark.parametrize("mode", ["vector", "auto"])
def test_sqlite_finds_and_ranks_as_the_in_memory_store_does(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    query: str,
    limit: int,
    filters: dict[str, Any],
) -> None:
    # Arrange: several reads in each search
    monkeypatch.setattr("pybodn._vectors.READ_BATCH", 7)
    with (
        MemoryEngine.in_memory(embedder=RANDOM) as memory,
        MemoryEngine.sqlite(tmp_path / "m.db", embedder=RANDOM) as sqlite_engine,
    ):
        fill(memory)
        fill(sqlite_engine)

        for namespace in NAMESPACES:
            # Act
            options: dict[str, Any] = {"mode": mode, "limit": limit, **filters}
            expected = memory.collection(namespace).search(query, **options)
            hits = sqlite_engine.collection(namespace).search(query, **options)

            # Assert
            assert expected, "every search finds records in every namespace"
            assert described(hits) == described(expected)
            assert [hit.score for hit in hits] == [hit.score for hit in expected]
