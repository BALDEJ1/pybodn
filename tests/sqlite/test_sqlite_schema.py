"""Opening a file: a new or empty one becomes pybodn's, an older one is brought up to date, and
anything else is refused and left exactly as it was."""

import sqlite3
import struct
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest

from pybodn import MemoryEngine
from pybodn._storage import schema, sqlite
from pybodn._words import WORD_RULES
from pybodn.errors import StorageError
from tests.sqlite.support import (
    RELEASE_AFTER,
    SHORT_WAIT,
    THREADS,
    WAIT_SECONDS,
    files_in,
    make_old_file,
    run_sql,
)


def make_pybodn_file(path: Path) -> None:
    with MemoryEngine.sqlite(path) as engine:
        engine.collection("user:123").add("Prefers Python")


def schema_objects(path: Path) -> list[tuple[object, ...]]:
    return run_sql(path, "SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY name")


def assert_is_pybodn_file(path: Path) -> None:
    assert run_sql(path, "PRAGMA application_id") == [(schema.APPLICATION_ID,)]
    assert run_sql(path, "PRAGMA user_version") == [(len(schema.MIGRATIONS),)]


@pytest.mark.parametrize(
    "prepare",
    [lambda path: None, lambda path: path.write_bytes(b"")],
    ids=["missing", "empty"],
)
def test_a_missing_or_empty_file_becomes_a_pybodn_file(
    path: Path, prepare: Callable[[Path], object]
) -> None:
    # Arrange
    prepare(path)

    # Act
    make_pybodn_file(path)

    # Assert
    assert_is_pybodn_file(path)


def test_each_version_keeps_its_layout(path: Path) -> None:
    # Files already written have exactly this layout: change it with a new migration instead.
    make_pybodn_file(path)

    columns = run_sql(path, "SELECT name, type, \"notnull\", pk FROM pragma_table_info('memories')")
    assert columns == [
        ("seq", "INTEGER", 0, 1),
        ("id", "TEXT", 1, 0),
        ("namespace", "TEXT", 1, 0),
        ("memory_key", "TEXT", 0, 0),
        ("content", "TEXT", 1, 0),
        ("metadata_json", "TEXT", 1, 0),
        ("tags_json", "TEXT", 1, 0),
        ("importance", "REAL", 1, 0),
        ("source", "TEXT", 0, 0),
        ("created_at", "TEXT", 1, 0),
        ("updated_at", "TEXT", 1, 0),
        ("version", "INTEGER", 1, 0),
        ("expires_at", "TEXT", 0, 0),
        ("archived_at", "TEXT", 0, 0),
    ]
    indexes = run_sql(
        path, "SELECT name, sql FROM sqlite_master WHERE type = 'index' ORDER BY name"
    )
    assert indexes == [
        (
            "ix_memories_expires",
            "CREATE INDEX ix_memories_expires ON memories (expires_at)"
            " WHERE expires_at IS NOT NULL",
        ),
        (
            "ix_memories_namespace_created",
            "CREATE INDEX ix_memories_namespace_created"
            " ON memories (namespace, created_at, archived_at, expires_at)",
        ),
        (
            "ix_memories_namespace_seq",
            "CREATE INDEX ix_memories_namespace_seq"
            " ON memories (namespace, seq, archived_at, expires_at)",
        ),
        (
            "ix_memory_vectors_model",
            "CREATE INDEX ix_memory_vectors_model ON memory_vectors (model)",
        ),
        (
            "sqlite_autoindex_memories_1",  # the one behind UNIQUE on id
            None,
        ),
        (
            "sqlite_autoindex_settings_1",  # the one behind the settings' primary key
            None,
        ),
        (
            "ux_memories_namespace_key",
            "CREATE UNIQUE INDEX ux_memories_namespace_key ON memories (namespace, memory_key) "
            "WHERE memory_key IS NOT NULL",
        ),
    ]
    settings = run_sql(
        path, 'SELECT name, type, "notnull", pk FROM pragma_table_info(?)', ("settings",)
    )
    assert settings == [("name", "TEXT", 0, 1), ("value", "", 1, 0)]
    vectors = run_sql(
        path, 'SELECT name, type, "notnull", pk FROM pragma_table_info(?)', ("memory_vectors",)
    )
    assert vectors == [
        ("seq", "INTEGER", 0, 1),
        ("record_id", "TEXT", 1, 0),
        ("model", "TEXT", 1, 0),
        ("vector", "BLOB", 1, 0),
    ]
    assert run_sql(path, "SELECT sql FROM sqlite_master WHERE name = 'memory_words'") == [
        ("CREATE VIRTUAL TABLE memory_words USING fts5(words, namespace_token, tokenize='ascii')",)
    ]


def test_an_older_file_is_brought_up_to_date(path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    with MemoryEngine.sqlite(path) as engine:
        record = engine.collection("user:123").add("Prefers Python")
    monkeypatch.setattr(schema, "MIGRATIONS", (*schema.MIGRATIONS, ("CREATE TABLE extra (x)",)))

    # Act
    with MemoryEngine.sqlite(path) as engine:
        stored = engine.collection("user:123").get(record.id)

    # Assert
    assert stored == record
    assert run_sql(path, "PRAGMA user_version") == [(len(schema.MIGRATIONS),)]
    assert run_sql(path, "SELECT name FROM sqlite_master WHERE name = 'extra'") == [("extra",)]


def fill_0_2_0(engine: MemoryEngine) -> None:
    notes = engine.collection("user:123")
    notes.add("Prefers Python", tags=["lang"])
    notes.put("drink", "Кава без цукру")


def test_a_file_from_pybodn_0_2_0_keeps_its_records_and_word_index(path: Path) -> None:
    # Arrange
    make_old_file(path, 2, fill_0_2_0)
    rows = run_sql(path, "SELECT * FROM memories ORDER BY seq")
    index = run_sql(path, "SELECT rowid, * FROM memory_words ORDER BY rowid")

    # Act
    with MemoryEngine.sqlite(path) as engine:
        records = engine.collection("user:123").list()

    # Assert
    assert [record.content for record in records] == ["Prefers Python", "Кава без цукру"]
    assert {(record.expires_at, record.archived_at) for record in records} == {(None, None)}
    assert run_sql(path, "SELECT * FROM memories ORDER BY seq") == [
        (*row, None, None) for row in rows
    ]
    assert run_sql(path, "SELECT rowid, * FROM memory_words ORDER BY rowid") == index
    assert_is_pybodn_file(path)


@pytest.mark.parametrize("version", range(1, len(schema.MIGRATIONS)))
def test_a_file_of_any_older_version_gains_the_layout_of_a_new_one(
    path: Path, tmp_path: Path, version: int
) -> None:
    # Arrange
    make_old_file(path, version, fill_0_2_0)
    new_file = tmp_path / "new.db"
    make_pybodn_file(new_file)

    # Act
    MemoryEngine.sqlite(path).close()

    # Assert
    assert schema_objects(path) == schema_objects(new_file)


def test_a_failed_migration_changes_nothing(path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    make_pybodn_file(path)
    before = schema_objects(path)
    latest = len(schema.MIGRATIONS)
    monkeypatch.setattr(
        schema, "MIGRATIONS", (*schema.MIGRATIONS, ("CREATE TABLE extra (x)", "NOT SQL"))
    )

    # Act / Assert
    with pytest.raises(StorageError) as caught:
        MemoryEngine.sqlite(path)
    assert isinstance(caught.value.__cause__, sqlite3.Error)
    assert run_sql(path, "PRAGMA user_version") == [(latest,)]
    assert schema_objects(path) == before


def test_engines_opening_a_new_file_at_once_create_it_once(path: Path) -> None:
    # Arrange
    start = threading.Barrier(THREADS, timeout=WAIT_SECONDS)

    def open_and_add() -> None:
        start.wait()
        with MemoryEngine.sqlite(path, timeout=WAIT_SECONDS) as engine:
            engine.collection("user:123").add("Prefers Python")

    # Act
    with ThreadPoolExecutor(THREADS) as pool:
        futures = [pool.submit(open_and_add) for _ in range(THREADS)]
    for future in futures:
        future.result()

    # Assert
    assert_is_pybodn_file(path)
    assert run_sql(path, "SELECT count(*) FROM memories") == [(THREADS,)]


# Each one leaves a file that is not pybodn's, or is too new for this version, at ``path``.


def text_file(path: Path) -> None:
    path.write_text("not a database")


def database_with_a_table(path: Path) -> None:
    run_sql(path, "CREATE TABLE notes (text)")


def database_with_a_view(path: Path) -> None:
    run_sql(path, "CREATE VIEW answer AS SELECT 42")


def database_of_another_program(path: Path) -> None:
    run_sql(path, "PRAGMA application_id = 42")


def database_with_a_version_only(path: Path) -> None:
    run_sql(path, "PRAGMA user_version = 1")


def pybodn_file_from_the_future(path: Path) -> None:
    make_pybodn_file(path)
    run_sql(path, f"PRAGMA user_version = {len(schema.MIGRATIONS) + 1}")


def pybodn_file_with_newer_word_rules(path: Path) -> None:
    make_pybodn_file(path)
    run_sql(path, "UPDATE settings SET value = ? WHERE name = 'word_rules'", (WORD_RULES + 1,))


def pybodn_file_with_a_trigger(path: Path) -> None:
    make_pybodn_file(path)
    run_sql(
        path,
        "CREATE TRIGGER overwrite AFTER INSERT ON memories"
        " BEGIN UPDATE memories SET content = 'overwritten'; END",
    )


def pybodn_file_with_a_view(path: Path) -> None:
    make_pybodn_file(path)
    run_sql(path, "CREATE VIEW everything AS SELECT * FROM memories")


def pybodn_file_whose_schema_calls_pybodn(path: Path) -> None:
    make_pybodn_file(path)
    with closing(sqlite3.connect(path)) as connection:
        connection.create_function("pybodn_words", 1, str, deterministic=True)
        connection.execute("CREATE INDEX costly ON memories (pybodn_words(content))")
        connection.commit()


REFUSED = [
    pytest.param(text_file, "file is not a database", id="text-file"),
    pytest.param(database_with_a_table, "not a pybodn database", id="table"),
    pytest.param(database_with_a_view, "not a pybodn database", id="view"),
    pytest.param(database_of_another_program, "not a pybodn database", id="another-program"),
    pytest.param(database_with_a_version_only, "not a pybodn database", id="version-only"),
    pytest.param(pybodn_file_from_the_future, "newer", id="newer-schema"),
    pytest.param(pybodn_file_with_newer_word_rules, "newer", id="newer-word-rules"),
    pytest.param(pybodn_file_with_a_trigger, "triggers or views", id="pybodn-with-trigger"),
    pytest.param(pybodn_file_with_a_view, "triggers or views", id="pybodn-with-view"),
    pytest.param(
        pybodn_file_whose_schema_calls_pybodn, "unsafe use", id="pybodn-function-in-schema"
    ),
]


@pytest.fixture
def remember_connections(monkeypatch: pytest.MonkeyPatch) -> Callable[[], list[sqlite3.Connection]]:
    """From the call on, every connection pybodn opens is added to the returned list."""

    def start() -> list[sqlite3.Connection]:
        opened: list[sqlite3.Connection] = []
        connect = sqlite3.connect

        def remembering_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
            connection: sqlite3.Connection = connect(*args, **kwargs)
            opened.append(connection)
            return connection

        monkeypatch.setattr(sqlite3, "connect", remembering_connect)
        return opened

    return start


def assert_closed(connections: list[sqlite3.Connection]) -> None:
    assert len(connections) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connections[0].execute("SELECT 1")


@pytest.mark.parametrize(("make", "message"), REFUSED)
def test_a_file_that_is_not_pybodns_or_is_too_new_is_refused_and_left_unchanged(
    path: Path,
    remember_connections: Callable[[], list[sqlite3.Connection]],
    make: Callable[[Path], None],
    message: str,
) -> None:
    # Arrange
    make(path)
    before = path.read_bytes()
    files = files_in(path.parent)
    opened = remember_connections()

    # Act / Assert
    with pytest.raises(StorageError, match=message):
        MemoryEngine.sqlite(path)
    assert_closed(opened)
    assert path.read_bytes() == before
    assert files_in(path.parent) == files


def test_a_failed_migration_closes_its_connection(
    path: Path,
    remember_connections: Callable[[], list[sqlite3.Connection]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    make_pybodn_file(path)
    monkeypatch.setattr(schema, "MIGRATIONS", (*schema.MIGRATIONS, ("NOT SQL",)))
    opened = remember_connections()

    # Act / Assert
    with pytest.raises(StorageError):
        MemoryEngine.sqlite(path)
    assert_closed(opened)


def test_an_interrupted_open_closes_its_connection(
    path: Path,
    remember_connections: Callable[[], list[sqlite3.Connection]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    def interrupt(*args: object) -> None:
        raise KeyboardInterrupt  # as Ctrl-C would, while the file is being opened

    monkeypatch.setattr(sqlite, "_use_write_ahead_log", interrupt)
    opened = remember_connections()

    # Act / Assert
    with pytest.raises(KeyboardInterrupt):
        MemoryEngine.sqlite(path)
    assert_closed(opened)


JOURNAL_MAGIC = bytes.fromhex("d9d505f920a163d7")


def plant_rollback_journal(path: Path, named: Path) -> None:
    """A rollback journal, as anyone who can create files beside ``path`` could leave it. Applied,
    it cuts the file to two pages and deletes ``named``, the file it gives as its super-journal."""
    header = bytearray(512)
    header[:8] = JOURNAL_MAGIC
    struct.pack_into(">III", header, 16, 2, 512, 4096)  # pages before, sector and page size
    name = str(named).encode()
    trailer = struct.pack(">II", len(name), sum(name)) + JOURNAL_MAGIC
    Path(f"{path}-journal").write_bytes(bytes(header) + name + trailer)


def test_a_rollback_journal_beside_the_file_is_refused_and_applied_to_nothing(
    path: Path, tmp_path: Path
) -> None:
    # Arrange
    make_pybodn_file(path)
    named = tmp_path / "elsewhere.txt"
    named.write_text("keep me")
    plant_rollback_journal(path, named)
    before = path.read_bytes()
    files = files_in(path.parent)

    # Act / Assert
    with pytest.raises(StorageError, match="journal"):
        MemoryEngine.sqlite(path, timeout=SHORT_WAIT)
    assert path.read_bytes() == before
    assert files_in(path.parent) == files
    assert named.read_text() == "keep me"


def test_opening_waits_for_a_rollback_journal_that_goes_away(path: Path) -> None:
    # Arrange: a journal there only for a moment, as while another process makes a new file
    make_pybodn_file(path)
    journal = Path(f"{path}-journal")
    journal.write_bytes(b"")
    removal = threading.Timer(RELEASE_AFTER, journal.unlink)
    removal.start()

    # Act
    with MemoryEngine.sqlite(path, timeout=WAIT_SECONDS) as engine:
        count = engine.collection("user:123").count()
    removal.join()

    # Assert
    assert count == 1
