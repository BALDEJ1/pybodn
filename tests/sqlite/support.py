"""What the SQLite tests share: timings, a moment long past, other connections to the file, as
another process would open them, files as older pybodn versions left them, and a record marked
in every field, whose traces the erase tests look for in the file."""

import os
import sqlite3
import sys
from collections.abc import Callable, Iterator, Mapping
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from multiprocessing.synchronize import Barrier
from pathlib import Path

import pytest

from pybodn import Embedder, MemoryEngine, MemoryRecord
from pybodn._storage import schema

SHORT_WAIT = 0.05  # seconds: enough for a free file, too short to wait out another writer
RELEASE_AFTER = 0.2  # seconds after which another connection frees the file
WAIT_SECONDS = 10  # seconds: generous for a slow machine, far below a wait that never ends
THREADS = 8
PAST = datetime(2000, 1, 1, tzinfo=UTC)  # an expiry long gone on the real clock

needs_read_only = pytest.mark.skipif(
    sys.platform == "win32" or os.geteuid() == 0,
    reason="Windows does not stop writes to a read-only folder, and neither does root",
)


def run_sql(path: Path, sql: str, parameters: tuple[object, ...] = ()) -> list[tuple[object, ...]]:
    """Run one statement on the file through a connection of its own."""
    with closing(sqlite3.connect(path, autocommit=True)) as connection:
        return connection.execute(sql, parameters).fetchall()


def format_marks(path: Path) -> tuple[object, ...]:
    """The application id, schema version and word rules of the file at ``path``."""
    return run_sql(
        path,
        "SELECT application_id, user_version, value FROM pragma_application_id,"
        " pragma_user_version, settings WHERE name = 'word_rules'",
    )[0]


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


# In no other text of these tests, and no other word starts with its first letter, so the word
# index keeps it whole.
MARK = "zqvmark"
SPACE = f"{MARK}space"
# The values of the marked record that the file holds as text.
TEXTS = {
    "text": f"{MARK}text",
    "key": f"{MARK}key",
    "tag": f"{MARK}tag",
    "field": f"{MARK}field",
    "value": f"{MARK}value",
    "source": f"{MARK}source",
    "namespace": SPACE,
}
# What a new version of the record no longer holds, and everything the record left.
OLD_VALUES = frozenset({"text", "tag", "field", "value", "source"})
EVERY_TRACE = OLD_VALUES | {"id", "key", "namespace", "token"}


def add_marked(engine: MemoryEngine) -> MemoryRecord:
    """A record with the mark in every field, alone in a namespace of its own."""
    return engine.collection(SPACE).add(
        TEXTS["text"],
        key=TEXTS["key"],
        tags=[TEXTS["tag"]],
        metadata={TEXTS["field"]: TEXTS["value"]},
        source=TEXTS["source"],
    )


def file_bytes(path: Path) -> bytes:
    """The database file and its write-ahead log, as one run of bytes."""
    log = path.with_name(f"{path.name}-wal")
    return path.read_bytes() + (log.read_bytes() if log.exists() else b"")


def left_in_file(path: Path, traces: Mapping[str, bytes]) -> set[str]:
    stored = file_bytes(path)
    return {name for name, trace in traces.items() if trace in stored}


def assert_in_file(path: Path, traces: Mapping[str, bytes]) -> None:
    """Every trace is in the file, so that a test cannot pass for want of them."""
    assert left_in_file(path, traces) == set(traces)


def traces_of(record: MemoryRecord) -> dict[str, bytes]:
    """The marked record's texts, its id and its namespace's token in the word index."""
    return {name: text.encode() for name, text in TEXTS.items()} | {
        "id": record.id.encode(),
        "token": SPACE.encode().hex().encode(),
    }


type Hide = Callable[[MemoryEngine, MemoryRecord], object]


def purge(engine: MemoryEngine, record: MemoryRecord) -> None:
    engine.collection(SPACE).update(record.id, expires_at=PAST)
    engine.purge_expired()


# Each way to hide the marked record or its old values, and the traces the erase removes.
# Archived records and expired ones not yet purged are still stored, so nothing goes.
HIDE: dict[str, tuple[Hide, frozenset[str]]] = {
    "delete": (lambda engine, record: engine.collection(SPACE).delete(record.id), EVERY_TRACE),
    "purge-expired": (purge, EVERY_TRACE),
    "replacing-put": (
        lambda engine, _: engine.collection(SPACE).put(TEXTS["key"], "plain words"),
        OLD_VALUES,
    ),
    "update": (
        lambda engine, record: engine.collection(SPACE).update(
            record.id, content="plain words", tags=[], metadata={}, source=None
        ),
        OLD_VALUES,
    ),
    "archive": (lambda engine, record: engine.collection(SPACE).archive(record.id), frozenset()),
    "expiry": (
        lambda engine, record: engine.collection(SPACE).update(record.id, expires_at=PAST),
        frozenset(),
    ),
}


def bytes_of(texts: list[str]) -> list[list[float]]:
    """A vector of each text's first eight bytes, so that each text has a vector of its own."""
    return [[float(byte) for byte in text.encode()[:8].ljust(8, b".")] for text in texts]


def stored_vector(path: Path, record: MemoryRecord) -> bytes:
    ((vector,),) = run_sql(
        path, "SELECT vector FROM memory_vectors WHERE record_id = ?", (record.id,)
    )
    assert isinstance(vector, bytes)
    return vector


# How another connection holds the file: a reader keeps the log from being emptied, and a writer
# keeps the erase from starting.
HOLD = {"reader": "BEGIN", "writer": "BEGIN IMMEDIATE"}
