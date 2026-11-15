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

from pybodn._filter import Filter
from pybodn._models import JSONScalar, MemoryRecord, SearchHit
from pybodn._search import SCORE_DECIMALS, Term
from pybodn._storage import schema
from pybodn._storage.base import Build, Lookup, ensure_open, reject_stored_keys
from pybodn._words import WORD_RULES, words
from pybodn.errors import StorageError

# The columns of a record; seq, the order of insertion, SQLite fills in itself. Every query
# is built from this list and the fixed pieces below: no value or outside text is ever
# pasted into one.
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
_INSERT = (
    f"INSERT INTO memories ({', '.join(_COLUMNS)})"  # noqa: S608
    f" VALUES ({', '.join(f':{column}' for column in _COLUMNS)})"
)
_REPLACE = (
    f"UPDATE memories SET {', '.join(f'{column} = :{column}' for column in _COLUMNS)}"  # noqa: S608
    " WHERE id = :current_id"
)
_COUNT = "SELECT count(*) FROM memories WHERE namespace = ?"
_PAGE = " ORDER BY seq LIMIT ? OFFSET ?"
_KEY_USED = "SELECT 1 FROM memories WHERE namespace = ? AND memory_key = ?"
_DELETE = "DELETE FROM memories WHERE namespace = ? AND id = ?"
_NO_LIMIT = -1  # SQLite's LIMIT for "all the rest"
_PAUSE = 0.01  # seconds between tries to switch to the write-ahead log
_BEGIN_WRITE = "BEGIN IMMEDIATE"  # takes the write lock at once, waiting up to the timeout
_BEGIN_READ = "BEGIN"  # every read until the end sees the file as it was at the first

# What each filter adds to a query's WHERE, with its values passed apart. A metadata field
# is found by its name, never by a path, and its JSON type keeps True apart from 1.
_HAS_TAG = " AND EXISTS (SELECT 1 FROM json_each(tags_json) WHERE value = ?)"
_FIELD_OF_TYPE = (  # for null, true and false the JSON type alone says the value
    " AND EXISTS (SELECT 1 FROM json_each(metadata_json) WHERE key = ? AND type = ?)"
)
_FIELD_IS_TEXT = (
    " AND EXISTS (SELECT 1 FROM json_each(metadata_json)"
    " WHERE key = ? AND type = 'text' AND value = ?)"
)
_FIELD_IS_NUMBER = (
    " AND EXISTS (SELECT 1 FROM json_each(metadata_json)"
    " WHERE key = ? AND type IN ('integer', 'real') AND value = ?)"
)
_MIN_IMPORTANCE = " AND importance >= ?"
_CREATED_SINCE = " AND created_at >= ?"
_CREATED_BEFORE = " AND created_at < ?"

# The word index holds each record's words and its namespace's token under its seq. The
# pybodn_ functions are Python functions this connection lends SQLite for these statements;
# nothing stored in the file calls them. OR REPLACE: a seq may still hold the words of a record
# that pybodn 0.1.0 deleted, and SQLite gives the seq of a deleted last record to the next one.
_INDEX_WORDS = (
    "INSERT OR REPLACE INTO memory_words (rowid, words, namespace_token)"
    " SELECT seq, pybodn_words(content), pybodn_namespace_token(namespace) FROM memories"
)
_ADD_WORDS = f"{_INDEX_WORDS} WHERE id = ?"
_DROP_WORDS = (
    "DELETE FROM memory_words"
    " WHERE rowid = (SELECT seq FROM memories WHERE namespace = ? AND id = ?)"
)
_SET_WORD_RULES = "INSERT OR REPLACE INTO settings (name, value) VALUES ('word_rules', ?)"
_VERSIONS = (
    "SELECT user_version, (SELECT value FROM settings WHERE name = 'word_rules')"
    " FROM pragma_user_version"
)

# The records a search finds, with their scores. The word index leads the join, so that FTS5
# finds the matches among its namespace's rows and weighs the terms once, and each record found
# is then read by its seq. The namespace's token weighs nothing in the score, and the score is
# rounded as the shared formula rounds it. The namespace is compared again, because one longer
# than 16 KB may share its token with another.
_SEARCH = (
    f"SELECT {', '.join(_COLUMNS)},"  # noqa: S608
    f" round(-bm25(memory_words, 1.0, 0.0), {SCORE_DECIMALS}) AS score"
    " FROM memory_words CROSS JOIN memories ON memories.seq = memory_words.rowid"
    " WHERE memory_words MATCH ? AND namespace = ?"
)
_BEST = " ORDER BY score DESC, seq LIMIT ?"


class SQLiteStore:
    """Records in an SQLite file, through one connection behind one lock.

    Every change takes the file's write lock first, so another process's change waits its
    turn, up to ``timeout`` seconds. Reads see every committed change at once.
    """

    __slots__ = ("_closed", "_connection", "_lock", "_versions")

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
                self._connection.create_function(
                    "pybodn_words", 1, _indexed_words, deterministic=True
                )
                self._connection.create_function(
                    "pybodn_namespace_token", 1, _namespace_token, deterministic=True
                )
                _require_features(self._connection)
                # two processes never migrate at once
                with _transaction(self._connection, _BEGIN_WRITE):
                    _migrate(self._connection)
                    _index_words(self._connection)
                    self._versions = _file_versions(self._connection)
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
            with self._up_to_date(_BEGIN_WRITE):
                reject_stored_keys(records, self._is_key_stored)
                self._connection.executemany(_INSERT, rows)
                self._connection.executemany(_ADD_WORDS, [(record.id,) for record in records])

    def modify(self, namespace: str, by: Lookup, value: str, build: Build) -> MemoryRecord:
        with self._lock:
            ensure_open(self)
            with self._up_to_date(_BEGIN_WRITE):
                current = self._find(namespace, by, value)
                record = build(current)
                if current is None:
                    self._connection.execute(_INSERT, _row(record))
                    self._connection.execute(_ADD_WORDS, (record.id,))
                else:
                    self._connection.execute(_REPLACE, {**_row(record), "current_id": current.id})
                    if record.content != current.content:  # replaces the old text's words
                        self._connection.execute(_ADD_WORDS, (record.id,))
        return record

    def delete(self, namespace: str, id: str) -> bool:
        with self._lock:
            ensure_open(self)
            with self._up_to_date(_BEGIN_WRITE):
                self._connection.execute(_DROP_WORDS, (namespace, id))  # while the row is there
                deleted = self._connection.execute(_DELETE, (namespace, id)).rowcount
        return deleted > 0

    def list(
        self, namespace: str, *, where: Filter, limit: int | None, offset: int
    ) -> builtins.list[MemoryRecord]:
        conditions, values = _conditions(where)
        sql = f"{_SELECT}{conditions}{_PAGE}"  # noqa: S608 - fixed pieces only
        page = (namespace, *values, _NO_LIMIT if limit is None else limit, offset)
        with self._lock:
            ensure_open(self)
            with _storage_errors():
                rows = self._connection.execute(sql, page).fetchall()
                return [_record(row) for row in rows]

    def count(self, namespace: str, *, where: Filter) -> int:
        conditions, values = _conditions(where)
        with self._lock:
            ensure_open(self)
            with _storage_errors():
                count: int = _single(self._connection, _COUNT + conditions, (namespace, *values))
        return count

    def search_words(
        self, namespace: str, terms: Sequence[Term], *, where: Filter, limit: int
    ) -> builtins.list[SearchHit]:
        conditions, values = _conditions(where)
        sql = f"{_SEARCH}{conditions}{_BEST}"  # noqa: S608 - fixed pieces only
        parameters = (_match(namespace, terms), namespace, *values, limit)
        with self._lock:
            ensure_open(self)
            # The index must follow the rules this process prepared the query by.
            with self._up_to_date(_BEGIN_READ):
                rows = self._connection.execute(sql, parameters).fetchall()
            return [SearchHit(record=_record(row), score=row["score"]) for row in rows]

    def close(self) -> None:
        with self._lock:
            self._closed = True
            with _storage_errors():
                self._connection.close()  # closing again does nothing

    @contextmanager
    def _up_to_date(self, begin: str) -> Iterator[None]:
        """A transaction on a file still at the versions this process opened.

        A newer pybodn may have upgraded the file since. This process would then write the old
        way, leaving the file partly up to date, or search an index built under other rules, so
        it stops instead.
        """
        with _storage_errors(), _transaction(self._connection, begin):
            if _file_versions(self._connection) != self._versions:
                raise StorageError(
                    "a newer pybodn has upgraded the file since this process opened it:"
                    " restart the process to write to it or search it"
                )
            yield

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
def _transaction(connection: sqlite3.Connection, begin: str) -> Iterator[None]:
    """Begin with ``begin``; commit at the end, or roll back on any error."""
    connection.execute(begin)
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


def _require_features(connection: sqlite3.Connection) -> None:
    """Refuse an SQLite that lacks what the file needs, before the file is touched."""
    try:
        connection.execute("SELECT json_valid('[]')").fetchall()
        connection.execute(f"CREATE VIRTUAL TABLE temp.trial USING {schema.WORD_INDEX}")
        connection.execute("DROP TABLE temp.trial")
    except sqlite3.OperationalError as error:
        raise StorageError(
            "pybodn needs SQLite with full-text search (FTS5) and JSON; this Python's SQLite"
            f" {sqlite3.sqlite_version} lacks one: {error}"
        ) from error


def _index_words(connection: sqlite3.Connection) -> None:
    """Index every record's words again when the file's index was built under other rules.

    A file from 0.1.0 has no index yet, and one built under older rules is out of date: both
    are filled under the current rules. Newer rules than this pybodn knows are refused.
    """
    _, rules = _file_versions(connection)
    if rules == WORD_RULES:
        return
    if isinstance(rules, int) and rules > WORD_RULES:
        raise StorageError(
            f"the file's word index follows rules {rules}, newer than the {WORD_RULES} this"
            " pybodn knows: upgrade pybodn to open it"
        )
    connection.execute("DELETE FROM memory_words")
    connection.execute(_INDEX_WORDS)
    connection.execute(_SET_WORD_RULES, (WORD_RULES,))


def _file_versions(connection: sqlite3.Connection) -> tuple[object, object]:
    """The file's schema version and the rules its word index was built with."""
    ((schema_version, rules),) = connection.execute(_VERSIONS).fetchall()
    return schema_version, rules


def _indexed_words(content: object) -> str:
    """A record's words as the index takes them: one space between each two."""
    return " ".join(words(content)) if isinstance(content, str) else ""  # damaged: no words


def _namespace_token(namespace: object) -> str:
    """The token for ``namespace`` in the word index: its UTF-8 bytes as hex digits.

    The tokenizer keeps hex digits together, so any namespace is one token. FTS5 cuts a token
    longer than 32 KB short, so namespaces longer than 16 KB may share one. Files keep the
    tokens they were indexed with: a change here must raise ``WORD_RULES``, so that they are
    indexed again.
    """
    return namespace.encode().hex() if isinstance(namespace, str) else ""  # damaged: no token


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


def _single(connection: sqlite3.Connection, sql: str, parameters: tuple[object, ...] = ()) -> Any:
    """The one value the query returns."""
    ((value,),) = connection.execute(sql, parameters).fetchall()
    return value


def _match(namespace: str, terms: Sequence[Term]) -> str:
    """An FTS5 query for the rows of ``namespace`` that any one of ``terms`` finds.

    Each term is a quoted string, ``"..."*`` for a prefix, so no word can be read as FTS5 syntax.
    """
    any_term = " OR ".join(
        '"' + term.text.replace('"', '""') + '"' + ("*" if term.is_prefix else "") for term in terms
    )
    return f'namespace_token : "{_namespace_token(namespace)}" AND words : ({any_term})'


def _conditions(where: Filter) -> tuple[str, tuple[object, ...]]:
    """What ``where`` adds to a query's WHERE, and the values it needs, in order."""
    sql: list[str] = []
    values: list[object] = []
    for tag in where.tags:
        sql.append(_HAS_TAG)
        values.append(tag)
    for key, value in where.metadata:
        piece, piece_values = _field_is(key, value)
        sql.append(piece)
        values.extend(piece_values)
    if where.min_importance is not None:
        sql.append(_MIN_IMPORTANCE)
        values.append(where.min_importance)
    if where.created_since is not None:
        sql.append(_CREATED_SINCE)
        values.append(_write_time(where.created_since))
    if where.created_before is not None:
        sql.append(_CREATED_BEFORE)
        values.append(_write_time(where.created_before))
    return "".join(sql), tuple(values)


def _field_is(key: str, value: JSONScalar) -> tuple[str, tuple[object, ...]]:
    """The condition that metadata field ``key`` equals ``value``, and the values it needs."""
    if value is None:
        return _FIELD_OF_TYPE, (key, "null")
    if isinstance(value, bool):
        return _FIELD_OF_TYPE, (key, "true" if value else "false")
    if isinstance(value, str):
        return _FIELD_IS_TEXT, (key, value)
    return _FIELD_IS_NUMBER, (key, value)


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
