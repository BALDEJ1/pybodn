"""What the SQLite tests share: timings, a moment long past, other connections to the file, as
another process would open them, and files as older pybodn versions left them."""

import sqlite3
from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from multiprocessing.synchronize import Barrier
from pathlib import Path

from pybodn import Embedder, MemoryEngine
from pybodn._storage import schema

SHORT_WAIT = 0.05  # seconds: enough for a free file, too short to wait out another writer
RELEASE_AFTER = 0.2  # seconds after which another connection frees the file
WAIT_SECONDS = 10  # seconds: generous for a slow machine, far below a wait that never ends
THREADS = 8
PAST = datetime(2000, 1, 1, tzinfo=UTC)  # an expiry long gone on the real clock


def run_sql(path: Path, sql: str, parameters: tuple[object, ...] = ()) -> list[tuple[object, ...]]:
    """Run one statement on the file through a connection of its own."""
    with closing(sqlite3.connect(path, autocommit=True)) as connection:
        return connection.execute(sql, parameters).fetchall()


def files_in(folder: Path) -> list[str]:
    return sorted(entry.name for entry in folder.iterdir())


def add_in_another_process(path: Path, namespace: str, contents: list[str], start: Barrier) -> None:
    """Wait at ``start`` for the other processes, then open the file and add each of
    ``contents`` in turn, one write each. Run in a process of its own."""
    start.wait(WAIT_SECONDS)
    with MemoryEngine.sqlite(path, timeout=WAIT_SECONDS) as engine:
        notes = engine.collection(namespace)
        for content in contents:
            notes.add(content)


@contextmanager
def another_writer(path: Path) -> Iterator[sqlite3.Connection]:
    """A connection of its own that holds the file's write lock until it commits, rolls back,
    or the block ends. Any thread may use it."""
    with closing(sqlite3.connect(path, autocommit=True, check_same_thread=False)) as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            yield connection
        finally:
            if connection.in_transaction:
                connection.execute("ROLLBACK")


def make_old_file(
    path: Path,
    version: int,
    fill: Callable[[MemoryEngine], object],
    *,
    embedder: Embedder | None = None,
) -> None:
    """A file at schema ``version``, as the pybodn of that version leaves it: the first
    ``version`` migrations, holding what ``fill`` writes today, with ``embedder`` if given, in
    every column and table that version has, and nothing that later versions add."""
    today = path.with_name(f"{path.name}.today")
    with MemoryEngine.sqlite(today, embedder=embedder) as engine:
        fill(engine)
    with closing(sqlite3.connect(path, autocommit=True)) as connection:
        connection.execute("ATTACH ? AS today", (str(today),))
        connection.execute("BEGIN")
        for statement in (sql for step in schema.MIGRATIONS[:version] for sql in step):
            connection.execute(statement)
        tables = connection.execute(
            "SELECT name FROM main.sqlite_master WHERE type = 'table'"
            " AND name NOT LIKE 'memory\\_words\\_%' ESCAPE '\\'"  # FTS5 keeps these itself
        ).fetchall()
        for (table,) in tables:
            names = [
                name
                for (name,) in connection.execute(
                    "SELECT name FROM pragma_table_info(?, 'main')", (table,)
                )
            ]
            columns = ", ".join(["rowid", *names] if table == "memory_words" else names)
            connection.execute(
                f"INSERT INTO main.{table} ({columns}) SELECT {columns} FROM today.{table}"  # noqa: S608
            )
        connection.execute(f"PRAGMA main.user_version = {version}")
        connection.execute(f"PRAGMA main.application_id = {schema.APPLICATION_ID}")
        connection.execute("COMMIT")
        connection.execute("DETACH today")
    for leftover in path.parent.glob(f"{today.name}*"):
        leftover.unlink()
