"""The word index in the file: every write keeps it exact, opening fills it again when it was
built under older rules, and a process that a newer pybodn has overtaken stops before it writes or
searches."""

import sqlite3
from collections.abc import Callable
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest

from pybodn import Collection, MemoryEngine, MemoryRecord, NewMemory
from pybodn._storage import schema
from pybodn._words import WORD_RULES, words
from pybodn.errors import StorageError
from tests.sqlite.support import run_sql


def query(path: Path, sql: str, parameters: tuple[object, ...] = ()) -> list[Any]:
    with closing(sqlite3.connect(path, autocommit=True)) as connection:
        connection.execute(
            "CREATE VIRTUAL TABLE temp.vocabulary USING fts5vocab(main, memory_words, instance)"
        )
        return connection.execute(sql, parameters).fetchall()


def indexed(path: Path, column: str) -> dict[int, list[str]]:
    """The tokens the word index holds in ``column``, by seq, in order."""
    held: dict[int, list[str]] = {}
    sql = "SELECT doc, term FROM temp.vocabulary WHERE col = ? ORDER BY doc, offset"
    for seq, token in query(path, sql, (column,)):
        held.setdefault(seq, []).append(token)
    return held


def assert_index_holds_every_record(path: Path) -> None:
    """The index holds each record's words and its namespace's token, and nothing else."""
    records = query(path, "SELECT seq, namespace, content FROM memories")
    assert indexed(path, "words") == {seq: words(text) for seq, _, text in records if words(text)}
    assert indexed(path, "namespace_token") == {
        seq: [namespace.encode().hex()] for seq, namespace, _ in records
    }


def index_data(path: Path) -> list[Any]:
    """Everything the word index has stored, as SQLite keeps it."""
    return query(path, "SELECT * FROM memory_words_data ORDER BY id")


def word_rules(path: Path) -> list[tuple[object, ...]]:
    return run_sql(path, "SELECT value FROM settings WHERE name = 'word_rules'")


def prepare(engine: MemoryEngine) -> tuple[Collection, Collection]:
    """Two namespaces with a few records; the last one added is in the first namespace."""
    notes, others = engine.collection("user:1"), engine.collection("user:2")
    others.add("Кава в іншому просторі")
    notes.add("Алергія на арахіс, п’є каву")
    notes.put("drink", "Кава без цукру")
    return notes, others


def first(collection: Collection) -> MemoryRecord:
    return collection.list()[0]


def last(collection: Collection) -> MemoryRecord:
    return collection.list()[-1]


# Each one changes the records ``prepare`` made, through one kind of write.
WRITES: dict[str, Callable[[Collection, Collection], object]] = {
    "add": lambda notes, others: notes.add("Чай з медом"),
    "add-many": lambda notes, others: notes.add_many(
        [NewMemory("Python"), NewMemory("Кава"), NewMemory("???")]
    ),
    "put-new": lambda notes, others: notes.put("tea", "Чай з медом"),
    "put-new-text": lambda notes, others: notes.put("drink", "Чай з медом"),
    "put-same-text": lambda notes, others: notes.put("drink", "Кава без цукру", tags=["x"]),
    "update-text": lambda notes, others: notes.update(first(notes).id, content="Горіхи"),
    "update-tags": lambda notes, others: notes.update(first(notes).id, tags=["health"]),
    "delete": lambda notes, others: notes.delete(first(notes).id),
    "delete-in-another-namespace": lambda notes, others: notes.delete(first(others).id),
    # SQLite gives the next record the seq of the last one deleted
    "delete-last-then-add": lambda notes, others: [
        notes.delete(last(notes).id),
        notes.add("Новий запис"),
    ],
}


@pytest.mark.parametrize("write", WRITES.values(), ids=WRITES.keys())
def test_every_write_keeps_the_word_index_exact(
    path: Path, write: Callable[[Collection, Collection], object]
) -> None:
    with MemoryEngine.sqlite(path) as engine:
        write(*prepare(engine))

    assert_index_holds_every_record(path)


@pytest.mark.parametrize("write", ["put-same-text", "update-tags"])
def test_a_write_that_keeps_the_text_leaves_the_word_index_alone(path: Path, write: str) -> None:
    with MemoryEngine.sqlite(path) as engine:
        # Arrange
        notes, others = prepare(engine)
        before = index_data(path)

        # Act
        WRITES[write](notes, others)

        # Assert
        assert index_data(path) == before


def test_words_that_pybodn_0_1_0_left_behind_do_not_block_the_next_record(path: Path) -> None:
    with MemoryEngine.sqlite(path) as engine:
        # Arrange: 0.1.0 deletes the last record but not its words
        notes, _ = prepare(engine)
        run_sql(path, "DELETE FROM memories WHERE seq = (SELECT max(seq) FROM memories)")

        # Act: SQLite gives that seq to the next record
        notes.add("Новий запис")

    # Assert
    assert_index_holds_every_record(path)


def make_0_1_0_file(path: Path) -> None:
    """A file as pybodn 0.1.0 leaves it: records, and nothing that schema version 2 adds."""
    with MemoryEngine.sqlite(path) as engine:
        prepare(engine)
    for sql in (
        "DROP TABLE memory_words",
        "DROP TABLE settings",
        "DROP TABLE memory_vectors",
        "DROP INDEX ix_memories_namespace_created",
        "PRAGMA user_version = 1",
    ):
        run_sql(path, sql)


def make_file_with_older_word_rules(path: Path) -> None:
    """A file whose word index was built under older rules: the words it holds are outdated,
    and some belong to a record that pybodn 0.1.0 has since deleted."""
    with MemoryEngine.sqlite(path) as engine:
        prepare(engine)
    run_sql(path, "UPDATE memories SET content = 'Слова за новими правилами'")
    run_sql(path, "DELETE FROM memories WHERE seq = (SELECT min(seq) FROM memories)")
    run_sql(path, "UPDATE settings SET value = ? WHERE name = 'word_rules'", (WORD_RULES - 1,))


@pytest.mark.parametrize("make", [make_0_1_0_file, make_file_with_older_word_rules])
def test_opening_a_file_without_an_up_to_date_word_index_fills_it(
    path: Path, make: Callable[[Path], None]
) -> None:
    # Arrange
    make(path)

    # Act
    MemoryEngine.sqlite(path).close()

    # Assert
    assert_index_holds_every_record(path)
    assert word_rules(path) == [(WORD_RULES,)]
    assert run_sql(path, "PRAGMA user_version") == [(len(schema.MIGRATIONS),)]


def test_opening_an_up_to_date_file_leaves_the_word_index_alone(path: Path) -> None:
    # Arrange
    with MemoryEngine.sqlite(path) as engine:
        prepare(engine)
    before = index_data(path)

    # Act
    MemoryEngine.sqlite(path).close()

    # Assert
    assert index_data(path) == before


@pytest.mark.parametrize(
    ("damaged", "column"),
    [("content", "words"), ("namespace", "namespace_token")],
    ids=["text", "namespace"],
)
def test_a_damaged_value_is_left_out_of_the_index_and_the_file_still_opens(
    path: Path, damaged: str, column: str
) -> None:
    # Arrange
    make_0_1_0_file(path)
    whole = run_sql(path, "SELECT seq FROM memories WHERE namespace = 'user:1' ORDER BY seq")
    run_sql(path, f"UPDATE memories SET {damaged} = x'00' WHERE namespace = 'user:2'")  # noqa: S608

    # Act
    MemoryEngine.sqlite(path).close()

    # Assert
    assert sorted(indexed(path, column)) == [seq for (seq,) in whole]


# What a newer pybodn leaves behind when it upgrades the file under a running process.
UPGRADES: dict[str, tuple[str, tuple[object, ...]]] = {
    "schema": (f"PRAGMA user_version = {len(schema.MIGRATIONS) + 1}", ()),
    "word-rules": ("UPDATE settings SET value = ? WHERE name = 'word_rules'", (WORD_RULES + 1,)),
}


@pytest.mark.parametrize("upgrade", UPGRADES.values(), ids=UPGRADES.keys())
@pytest.mark.parametrize("write", WRITES.values(), ids=WRITES.keys())
def test_a_process_overtaken_by_a_newer_pybodn_stops_before_its_next_write(
    path: Path,
    upgrade: tuple[str, tuple[object, ...]],
    write: Callable[[Collection, Collection], object],
) -> None:
    with MemoryEngine.sqlite(path) as engine:
        # Arrange
        notes, others = prepare(engine)
        before = run_sql(path, "SELECT * FROM memories ORDER BY seq")
        run_sql(path, *upgrade)

        # Act / Assert
        with pytest.raises(StorageError, match="restart"):
            write(notes, others)
        assert run_sql(path, "SELECT * FROM memories ORDER BY seq") == before
        assert len(notes.list()) == 2  # reading still works


@pytest.mark.parametrize("upgrade", UPGRADES.values(), ids=UPGRADES.keys())
def test_a_process_overtaken_by_a_newer_pybodn_stops_before_it_searches(
    path: Path, upgrade: tuple[str, tuple[object, ...]]
) -> None:
    with MemoryEngine.sqlite(path) as engine:
        # Arrange: the index may now follow other rules than this process prepares the query by
        notes, _ = prepare(engine)
        run_sql(path, *upgrade)

        # Act / Assert
        with pytest.raises(StorageError, match="restart"):
            notes.search("кава")


def test_an_sqlite_without_the_word_index_it_needs_is_refused_before_the_file_changes(
    path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    make_0_1_0_file(path)
    before = path.read_bytes()
    monkeypatch.setattr(schema, "WORD_INDEX", "fts5(words, tokenize='no_such_tokenizer')")

    # Act / Assert
    with pytest.raises(StorageError, match="needs SQLite with full-text search") as caught:
        MemoryEngine.sqlite(path)
    assert isinstance(caught.value.__cause__, sqlite3.Error)
    assert path.read_bytes() == before
