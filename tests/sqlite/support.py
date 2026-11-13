"""What the SQLite tests share: timings, and other connections to the file, as another process
would open them."""

import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path

SHORT_WAIT = 0.05  # seconds: enough for a free file, too short to wait out another writer
RELEASE_AFTER = 0.2  # seconds after which another connection frees the file
WAIT_SECONDS = 10  # seconds: generous for a slow machine, far below a wait that never ends
THREADS = 8


def run_sql(path: Path, sql: str) -> list[tuple[object, ...]]:
    """Run one statement on the file through a connection of its own."""
    with closing(sqlite3.connect(path, autocommit=True)) as connection:
        return connection.execute(sql).fetchall()


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
