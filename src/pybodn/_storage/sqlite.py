"""A store that keeps records in an SQLite database file."""

import builtins
import json
import sqlite3
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

from pybodn._models import MemoryRecord
from pybodn._storage import schema
from pybodn._storage.base import Build, Lookup, ensure_open, reject_stored_keys
from pybodn.errors import StorageError

# The columns of a record; seq, the order of insertion, SQLite fills in itself. Every query
# is built from this list once, here: no value or outside text is ever pasted into one.
_COLUMNS = (
    "id",
    "namespace",
    "memory_key",
    "content",
    "metadata_json",
    "tags_json",
    "importance",
    "source",
    "created_at",
    "updated_at",
    "version",
)
_SELECT = f"SELECT {', '.join(_COLUMNS)} FROM memories WHERE namespace = ?"  # noqa: S608
_SELECT_BY: dict[Lookup, str] = {
    "id": f"{_SELECT} AND id = ?",
    "key": f"{_SELECT} AND memory_key = ?",
}
_SELECT_PAGE = f"{_SELECT} ORDER BY seq LIMIT ? OFFSET ?"
_INSERT = (
    f"INSERT INTO memories ({', '.join(_COLUMNS)})"  # noqa: S608
    f" VALUES ({', '.join(f':{column}' for column in _COLUMNS)})"
)
_REPLACE = (
    f"UPDATE memories SET {', '.join(f'{column} = :{column}' for column in _COLUMNS)}"  # noqa: S608
    " WHERE id = :current_id"
)
_COUNT = "SELECT count(*) FROM memories WHERE namespace = ?"
_KEY_USED = "SELECT 1 FROM memories WHERE namespace = ? AND memory_key = ?"
_DELETE = "DELETE FROM memories WHERE namespace = ? AND id = ?"
_NO_LIMIT = -1  # SQLite's LIMIT for "all the rest"
_PAUSE = 0.01  # seconds between tries to switch to the write-ahead log


class SQLiteStore:
    """Records in an SQLite file, through one connection behind one lock.

    Every change takes the file's write lock first, so another process's change waits its
    turn, up to ``timeout`` seconds. Reads see every committed change at once.
    """

    __slots__ = ("_closed", "_connection", "_lock")

    def __init__(self, path: str, *, timeout: float) -> None:
        """Open the file at ``path``, made if missing; ``":memory:"`` keeps records in memory.

        A file that is not pybodn's, or is from a newer pybodn, raises ``StorageError`` and is
        left as it was.
        """
        self._lock = threading.Lock()
        self._closed = False
        with _storage_errors():
            self._connection = sqlite3.connect(
                path, timeout=timeout, autocommit=True, check_same_thread=False
            )
        self._connection.row_factory = sqlite3.Row
        try:
            with _storage_errors():
                with _transaction(self._connection):  # two processes never migrate at once
                    _migrate(self._connection)
                _use_write_ahead_log(self._connection, timeout)
                # Each commit is flushed to disk before the call returns.
                self._connection.execute("PRAGMA synchronous = FULL").fetchall()
        except BaseException:
            self._connection.close()
            raise

    @property
    def closed(self) -> bool:
        return self._closed

    def get(self, namespace: str, by: Lookup, value: str) -> MemoryRecord | None:
        with self._lock:
            ensure_open(self)
            with _storage_errors():
                return self._find(namespace, by, value)

    def insert(self, records: Sequence[MemoryRecord]) -> None:
        rows = [_row(record) for record in records]
        with self._lock:
            ensure_open(self)
            if not rows:
                return
            with _storage_errors(), _transaction(self._connection):
                reject_stored_keys(records, self._is_key_stored)
                self._connection.executemany(_INSERT, rows)

    def modify(self, namespace: str, by: Lookup, value: str, build: Build) -> MemoryRecord:
        with self._lock:
            ensure_open(self)
            with _storage_errors(), _transaction(self._connection):
                current = self._find(namespace, by, value)
                record = build(current)
                if current is None:
                    self._connection.execute(_INSERT, _row(record))
                else:
                    self._connection.execute(_REPLACE, {**_row(record), "current_id": current.id})
        return record

    def delete(self, namespace: str, id: str) -> bool:
        with self._lock:
            ensure_open(self)
            with _storage_errors(), _transaction(self._connection):
                deleted = self._connection.execute(_DELETE, (namespace, id)).rowcount
        return deleted > 0

    def list(
        self, namespace: str, *, limit: int | None, offset: int
    ) -> builtins.list[MemoryRecord]:
        page = (namespace, _NO_LIMIT if limit is None else limit, offset)
        with self._lock:
            ensure_open(self)
            with _storage_errors():
                rows = self._connection.execute(_SELECT_PAGE, page).fetchall()
                return [_record(row) for row in rows]

    def count(self, namespace: str) -> int:
        with self._lock:
            ensure_open(self)
            with _storage_errors():
                count: int = _single(self._connection, _COUNT, (namespace,))
        return count

    def close(self) -> None:
        with self._lock:
            self._closed = True
            with _storage_errors():
                self._connection.close()  # closing again does nothing

    def _find(self, namespace: str, by: Lookup, value: str) -> MemoryRecord | None:
        rows = self._connection.execute(_SELECT_BY[by], (namespace, value)).fetchall()
        return _record(rows[0]) if rows else None

    def _is_key_stored(self, namespace: str, key: str) -> bool:
        return bool(self._connection.execute(_KEY_USED, (namespace, key)).fetchall())


@contextmanager
def _storage_errors() -> Iterator[None]:
    """Turn a failure of SQLite into ``StorageError``, with the original as its cause."""
    try:
        yield
    except sqlite3.Error as error:
        raise StorageError(f"the database failed: {error}") from error


@contextmanager
def _transaction(connection: sqlite3.Connection) -> Iterator[None]:
    """Take the write lock, waiting up to the timeout; commit, or roll back on any error."""
    connection.execute("BEGIN IMMEDIATE")
    try:
        yield
        connection.execute("COMMIT")
    except BaseException:
        if connection.in_transaction:  # a failed COMMIT may have rolled back already
            connection.execute("ROLLBACK")
        raise


def _migrate(connection: sqlite3.Connection) -> None:
    """Bring the file to the latest version of the schema, or refuse it unchanged."""
    application_id = _single(connection, "PRAGMA application_id")
    version: int = _single(connection, "PRAGMA user_version")
    is_new = application_id == 0 and version == 0
    is_empty = _single(connection, "SELECT count(*) FROM sqlite_master") == 0
    if application_id != schema.APPLICATION_ID and not (is_new and is_empty):
        raise StorageError("the file is not a pybodn database")
    latest = len(schema.MIGRATIONS)
    if version > latest:
        raise StorageError(
            f"the file has schema version {version}, newer than the {latest} this pybodn"
            " knows: upgrade pybodn to open it"
        )
    for number in range(version + 1, latest + 1):
        for statement in schema.MIGRATIONS[number - 1]:
            connection.execute(statement)
        connection.execute(f"PRAGMA user_version = {number}")
    if is_new:
        connection.execute(f"PRAGMA application_id = {schema.APPLICATION_ID}")


def _use_write_ahead_log(connection: sqlite3.Connection, timeout: float) -> None:
    """Switch the file to the write-ahead log, so that readers never block the writer.

    While another connection writes, SQLite refuses the switch at once rather than risk a
    deadlock, so retry here, up to ``timeout``.
    """
    deadline = time.monotonic() + timeout
    while True:
        try:
            connection.execute("PRAGMA journal_mode = WAL").fetchall()
        except sqlite3.OperationalError as error:
            if error.sqlite_errorcode != sqlite3.SQLITE_BUSY or time.monotonic() >= deadline:
                raise
            time.sleep(_PAUSE)
        else:
            return


def _single(connection: sqlite3.Connection, sql: str, parameters: tuple[str, ...] = ()) -> Any:
    """The one value the query returns."""
    ((value,),) = connection.execute(sql, parameters).fetchall()
    return value


def _row(record: MemoryRecord) -> dict[str, object]:
    """``record`` as the value of each column."""
    return {
        "id": record.id,
        "namespace": record.namespace,
        "memory_key": record.key,
        "content": record.content,
        "metadata_json": _json(record.metadata),
        "tags_json": _json(record.tags),
        "importance": record.importance,
        "source": record.source,
        "created_at": _write_time(record.created_at),
        "updated_at": _write_time(record.updated_at),
        "version": record.version,
    }


def _record(row: sqlite3.Row) -> MemoryRecord:
    """A row of the table as a record. A row that no longer reads back raises ``StorageError``."""
    try:
        return MemoryRecord(
            id=row["id"],
            namespace=row["namespace"],
            content=row["content"],
            key=row["memory_key"],
            metadata=json.loads(row["metadata_json"]),
            tags=tuple(json.loads(row["tags_json"])),
            importance=row["importance"],
            source=row["source"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            version=row["version"],
        )
    except (TypeError, ValueError, RecursionError) as error:
        raise StorageError(f"record {row['id']!r} in the database is damaged: {error}") from error


def _json(value: object) -> str:
    # Keys stay in the order given, as the in-memory store keeps them.
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _write_time(moment: datetime) -> str:
    """UTC as text of one width, microseconds always written, so text order is time order."""
    return moment.astimezone(UTC).isoformat(timespec="microseconds")
