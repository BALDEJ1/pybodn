"""Opening a file: a new or empty one becomes pybodn's, an older one is brought up to date, and
anything else is refused and left exactly as it was."""

import sqlite3
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest

from pybodn import MemoryEngine
from pybodn._storage import schema
from pybodn.errors import StorageError

type RunSQL = Callable[[str], list[tuple[object, ...]]]

THREADS = 8
WAIT_SECONDS = 10


def make_pybodn_file(path: Path) -> None:
    with MemoryEngine.sqlite(path) as engine:
        engine.collection("user:123").add("Prefers Python")


def make_database(path: Path, sql: str) -> None:
    with closing(sqlite3.connect(path, autocommit=True)) as connection:
        connection.execute(sql)


def schema_objects(run_sql: RunSQL) -> list[tuple[object, ...]]:
    return run_sql("SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY name")


def files_in(folder: Path) -> list[str]:
    return sorted(entry.name for entry in folder.iterdir())


def assert_is_pybodn_file(run_sql: RunSQL) -> None:
    assert run_sql("PRAGMA application_id") == [(schema.APPLICATION_ID,)]
    assert run_sql("PRAGMA user_version") == [(len(schema.MIGRATIONS),)]


def test_a_missing_file_is_created_as_a_pybodn_file(path: Path, run_sql: RunSQL) -> None:
    # Act
    with MemoryEngine.sqlite(path):
        pass

    # Assert
    assert_is_pybodn_file(run_sql)


def test_an_empty_file_becomes_a_pybodn_file(path: Path, run_sql: RunSQL) -> None:
    # Arrange
    path.write_bytes(b"")

    # Act
    with MemoryEngine.sqlite(path) as engine:
        engine.collection("user:123").add("Prefers Python")

    # Assert
    assert_is_pybodn_file(run_sql)


def test_version_1_keeps_its_layout(path: Path, run_sql: RunSQL) -> None:
    # Files already written have exactly this layout: change it with a new migration instead.
    make_pybodn_file(path)

    columns = run_sql("SELECT name, type, \"notnull\", pk FROM pragma_table_info('memories')")
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
    ]
    indexes = run_sql("SELECT name, sql FROM sqlite_master WHERE type = 'index' ORDER BY name")
    assert indexes == [
        (
            "ix_memories_namespace_seq",
            "CREATE INDEX ix_memories_namespace_seq ON memories (namespace, seq)",
        ),
        (
            "sqlite_autoindex_memories_1",  # the one behind UNIQUE on id
            None,
        ),
        (
            "ux_memories_namespace_key",
            "CREATE UNIQUE INDEX ux_memories_namespace_key ON memories (namespace, memory_key) "
            "WHERE memory_key IS NOT NULL",
        ),
    ]


def test_an_older_file_is_brought_up_to_date(
    path: Path, run_sql: RunSQL, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    with MemoryEngine.sqlite(path) as engine:
        record = engine.collection("user:123").add("Prefers Python")
    monkeypatch.setattr(schema, "MIGRATIONS", (*schema.MIGRATIONS, ("CREATE TABLE extra (x)",)))

    # Act
    with MemoryEngine.sqlite(path) as engine:
        stored = engine.collection("user:123").get(record.id)

    # Assert
    assert stored == record
    assert run_sql("PRAGMA user_version") == [(2,)]
    assert run_sql("SELECT name FROM sqlite_master WHERE name = 'extra'") == [("extra",)]


def test_a_failed_migration_changes_nothing(
    path: Path, run_sql: RunSQL, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    make_pybodn_file(path)
    before = schema_objects(run_sql)
    monkeypatch.setattr(
        schema, "MIGRATIONS", (*schema.MIGRATIONS, ("CREATE TABLE extra (x)", "NOT SQL"))
    )

    # Act / Assert
    with pytest.raises(StorageError) as caught:
        MemoryEngine.sqlite(path)
    assert isinstance(caught.value.__cause__, sqlite3.Error)
    assert run_sql("PRAGMA user_version") == [(1,)]
    assert schema_objects(run_sql) == before


def test_engines_opening_a_new_file_at_once_create_it_once(path: Path, run_sql: RunSQL) -> None:
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
    assert_is_pybodn_file(run_sql)
    assert run_sql("SELECT count(*) FROM memories") == [(THREADS,)]


# Each one leaves a file that is not pybodn's, or is too new for this version, at ``path``.


def text_file(path: Path) -> None:
    path.write_text("not a database")


def database_with_a_table(path: Path) -> None:
    make_database(path, "CREATE TABLE notes (text)")


def database_with_a_view(path: Path) -> None:
    make_database(path, "CREATE VIEW answer AS SELECT 42")


def database_of_another_program(path: Path) -> None:
    make_database(path, "PRAGMA application_id = 42")


def database_with_a_version_only(path: Path) -> None:
    make_database(path, "PRAGMA user_version = 1")


def pybodn_file_from_the_future(path: Path) -> None:
    make_pybodn_file(path)
    make_database(path, f"PRAGMA user_version = {len(schema.MIGRATIONS) + 1}")


REFUSED = [
    pytest.param(text_file, "file is not a database", id="text-file"),
    pytest.param(database_with_a_table, "not a pybodn database", id="table"),
    pytest.param(database_with_a_view, "not a pybodn database", id="view"),
    pytest.param(database_of_another_program, "not a pybodn database", id="another-program"),
    pytest.param(database_with_a_version_only, "not a pybodn database", id="version-only"),
    pytest.param(pybodn_file_from_the_future, "newer", id="newer-schema"),
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
