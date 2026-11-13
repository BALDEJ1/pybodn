"""A store that keeps records in an SQLite database file."""

import builtins
import json
import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

from pybodn._models import MemoryRecord
from pybodn._storage import schema
from pybodn._storage.base import Build, Lookup, ensure_open
from pybodn.errors import DuplicateKeyError, StorageError

# Every query is written out in full: no value and no column name is ever pasted into one.
_SELECT_BY: dict[Lookup, str] = {
    "id": "SELECT id, namespace, memory_key, content, metadata_json, tags_json, importance,"
    " source, created_at, updated_at, version FROM memories WHERE namespace = ? AND id = ?",
    "key": "SELECT id, namespace, memory_key, content, metadata_json, tags_json, importance,"
    " source, created_at, updated_at, version FROM memories"
    " WHERE namespace = ? AND memory_key = ?",
}
_SELECT_PAGE = (
    "SELECT id, namespace, memory_key, content, metadata_json, tags_json, importance, source,"
    " created_at, updated_at, version FROM memories WHERE namespace = ?"
    " ORDER BY seq LIMIT ? OFFSET ?"
)
_COUNT = "SELECT count(*) FROM memories WHERE namespace = ?"
_KEY_USED = "SELECT 1 FROM memories WHERE namespace = ? AND memory_key = ?"
_INSERT = (
    "INSERT INTO memories (id, namespace, memory_key, content, metadata_json, tags_json,"
    " importance, source, created_at, updated_at, version) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)
_REPLACE = (
    "UPDATE memories SET content = ?, metadata_json = ?, tags_json = ?, importance = ?,"
    " source = ?, created_at = ?, updated_at = ?, version = ? WHERE id = ?"
)
_DELETE = "DELETE FROM memories WHERE namespace = ? AND id = ?"
_NO_LIMIT = -1  # SQLite's LIMIT for "all the rest"


class SQLiteStore:
    """Records in an SQLite file, through one connection behind one lock, so threads can share it.

    Every change is one transaction that takes the file's write lock first, so a change from
    another process waits its turn, up to ``timeout`` seconds. Reads open no transaction of
    their own, so they always see the latest committed change, whoever made it.
    """

    __slots__ = ("_closed", "_connection", "_lock")

    def __init__(self, path: str, *, timeout: float) -> None:
        """Open the file at ``path``, made if missing; ``":memory:"`` keeps records in memory.

        A file that is not pybodn's, or was written by a newer pybodn, raises ``StorageError``
        and is left as it was.
        """
        self._lock = threading.Lock()
        self._closed = False
        with _storage_errors():
            self._connection = sqlite3.connect(
                path, timeout=timeout, autocommit=True, check_same_thread=False
            )
        try:
            with _storage_errors():
                with _transaction(self._connection):  # two processes never migrate at once
                    _migrate(self._connection)
                # Readers then never block the writer, and a commit survives a power cut.
                self._connection.execute("PRAGMA journal_mode = WAL").fetchall()
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
                rows = self._connection.execute(_SELECT_BY[by], (namespace, value)).fetchall()
        return _record(rows[0]) if rows else None

    def insert(self, records: Sequence[MemoryRecord]) -> None:
        rows = [_row(record) for record in records]
        with self._lock:
            ensure_open(self)
            if not rows:
                return
            with _storage_errors(), _transaction(self._connection):
                used: set[tuple[str, str]] = set()
                for record in records:
                    if record.key is None:
                        continue
                    slot = (record.namespace, record.key)
                    if slot in used or self._connection.execute(_KEY_USED, slot).fetchall():
                        raise DuplicateKeyError(
                            f"key {record.key!r} is already used in namespace {record.namespace!r}"
                        )
                    used.add(slot)
                self._connection.executemany(_INSERT, rows)

    def modify(self, namespace: str, by: Lookup, value: str, build: Build) -> MemoryRecord:
        with self._lock:
            ensure_open(self)
            with _storage_errors(), _transaction(self._connection):
                rows = self._connection.execute(_SELECT_BY[by], (namespace, value)).fetchall()
                current = _record(rows[0]) if rows else None
                record = build(current)
                if current is None:
                    self._connection.execute(_INSERT, _row(record))
                else:
                    self._connection.execute(_REPLACE, _replacement(record))
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


def _single(connection: sqlite3.Connection, sql: str, parameters: tuple[str, ...] = ()) -> Any:
    """The one value the query returns."""
    ((value,),) = connection.execute(sql, parameters).fetchall()
    return value


def _row(record: MemoryRecord) -> tuple[object, ...]:
    """``record`` as the values of ``_INSERT``."""
    return (
        record.id,
        record.namespace,
        record.key,
        record.content,
        _json(record.metadata),
        _json(builtins.list(record.tags)),
        record.importance,
        record.source,
        _write_time(record.created_at),
        _write_time(record.updated_at),
        record.version,
    )


def _replacement(record: MemoryRecord) -> tuple[object, ...]:
    """``record`` as the values of ``_REPLACE``: everything but its id, namespace and key."""
    return (
        record.content,
        _json(record.metadata),
        _json(builtins.list(record.tags)),
        record.importance,
        record.source,
        _write_time(record.created_at),
        _write_time(record.updated_at),
        record.version,
        record.id,
    )


def _record(row: tuple[Any, ...]) -> MemoryRecord:
    """A row of ``_SELECT_BY`` or ``_SELECT_PAGE`` as a record."""
    (
        record_id,
        namespace,
        key,
        content,
        metadata,
        tags,
        importance,
        source,
        created,
        updated,
        version,
    ) = row
    return MemoryRecord(
        id=record_id,
        namespace=namespace,
        content=content,
        key=key,
        metadata=json.loads(metadata),
        tags=tuple(json.loads(tags)),
        importance=importance,
        source=source,
        created_at=_read_time(created),
        updated_at=_read_time(updated),
        version=version,
    )


def _json(value: object) -> str:
    # Keys stay in the order given, as the in-memory store keeps them.
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _write_time(moment: datetime) -> str:
    """UTC as text of one width, microseconds always written, so text order is time order."""
    return moment.astimezone(UTC).isoformat(timespec="microseconds")


def _read_time(text: str) -> datetime:
    return datetime.fromisoformat(text)
