"""A store that keeps records in an SQLite database file."""

import builtins
import json
import sqlite3
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

from pybodn._filter import Filter
from pybodn._models import JSONScalar, MemoryRecord, SearchHit
from pybodn._search import SCORE_DECIMALS, Term
from pybodn._storage import schema
from pybodn._storage.base import (
    NO_VECTORS,
    Build,
    Lookup,
    MissingVector,
    TextAndKey,
    Unchanged,
    Vector,
    check_size,
    check_sizes,
    ensure_open,
    reject_stored_keys,
    split_batch,
    text_and_key,
    vector_after,
)
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
    "expires_at",
    "archived_at",
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
_NOT_ARCHIVED = " AND archived_at IS NULL"
_EXPIRED = "expires_at <= ?"  # as pybodn._filter.is_expired: NULL never expires
_NOT_EXPIRED = f" AND (expires_at IS NULL OR NOT ({_EXPIRED}))"

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

# Vectors, under the seq of their record. OR REPLACE: a seq may still hold the vector of a record
# that pybodn 0.1.0 deleted, as with words. A vector is given only if its record still holds the
# text it was made from.
_SET_VECTOR = (
    "INSERT OR REPLACE INTO memory_vectors (seq, record_id, model, vector)"
    " SELECT seq, id, ?, ? FROM memories WHERE id = ?"
)
_GIVE_VECTOR = f"{_SET_VECTOR} AND content = ?"
_DROP_VECTOR = (
    "DELETE FROM memory_vectors"
    " WHERE seq = (SELECT seq FROM memories WHERE namespace = ? AND id = ?)"
)
_VECTOR_SIZE = "SELECT length(vector) FROM memory_vectors WHERE model = ? LIMIT 1"
_LAST_SEQ = "SELECT coalesce(max(seq), 0) FROM memories"
_MISSING_VECTORS = (
    "SELECT seq, namespace, id, content FROM memories WHERE seq > ? AND seq <= ?"
    " AND NOT EXISTS (SELECT 1 FROM memory_vectors WHERE memory_vectors.seq = memories.seq"
    " AND memory_vectors.record_id = memories.id AND model = ?)"
    " ORDER BY seq LIMIT ?"
)

# The vectors of one model in one namespace, by seq. The namespace's records lead the join, in
# the order added, and each one's vector is then read by its seq; a vector another record left
# at that seq is passed over. The records a search finds are read afterwards, by seq.
_SEARCH_VECTORS = (
    "SELECT memories.seq, vector FROM memories CROSS JOIN memory_vectors"
    " ON memory_vectors.seq = memories.seq AND memory_vectors.record_id = memories.id"
    " WHERE namespace = ? AND model = ?"
)
_IN_ORDER = " ORDER BY memories.seq"
_RECORDS_AT = (
    f"SELECT seq, {', '.join(_COLUMNS)} FROM memories"  # noqa: S608
    " WHERE seq IN (SELECT value FROM json_each(?))"
)

# The copies of a batch: the first record added of each text and key, among the records of a
# namespace that hold a text of the batch, and a key of it or none, and that the filter keeps.
# The batch's texts and keys come as JSON, which older SQLite, such as 3.34, cuts at a NUL
# character: a batch that holds one compares them by the hex digits of their UTF-8 bytes
# instead, which takes longer, since every record's text is turned into them.
_COPIES = (
    f"SELECT {', '.join(_COLUMNS)} FROM memories WHERE seq IN (SELECT min(seq) FROM memories"  # noqa: S608
    " WHERE namespace = ?"
)
_AS_WRITTEN = (
    " AND content IN (SELECT value FROM json_each(?))"
    " AND (memory_key IS NULL OR memory_key IN (SELECT value FROM json_each(?)))"
)
_AS_BYTES = (
    " AND hex(content) IN (SELECT value FROM json_each(?))"
    " AND (memory_key IS NULL OR hex(memory_key) IN (SELECT value FROM json_each(?)))"
)
_FIRST_OF_EACH = " GROUP BY content, memory_key)"
_NUL = "\x00"

# A purge: the seqs of the records that have expired, through the expiry index, then their words,
# vectors and rows, each found by a seq of that list.
_EXPIRED_SEQS = f"SELECT seq FROM memories WHERE {_EXPIRED} LIMIT ?"  # noqa: S608
_PURGE = (
    "DELETE FROM memory_words WHERE rowid IN (SELECT value FROM json_each(?))",
    "DELETE FROM memory_vectors WHERE seq IN (SELECT value FROM json_each(?))",
    "DELETE FROM memories WHERE seq IN (SELECT value FROM json_each(?))",
)


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
                # Each commit is flushed to disk before the call returns, from the first one on.
                # This is also the first read of the file, so a file SQLite cannot use, such as
                # one in a read-only folder, fails here with SQLite's own reason.
                self._connection.execute("PRAGMA synchronous = FULL").fetchall()
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
        except BaseException:
            self._connection.close()
            raise

    @property
    def closed(self) -> bool:
        return self._closed

    def get(self, namespace: str, by: Lookup, value: str) -> MemoryRecord | None:
        with self._lock:
            ensure_open(self)
            with self._up_to_date(_BEGIN_READ):
                return self._find(namespace, by, value)

    def find_copies(
        self, namespace: str, records: Sequence[MemoryRecord], *, where: Filter
    ) -> dict[TextAndKey, MemoryRecord]:
        with self._lock:
            ensure_open(self)
            with self._up_to_date(_BEGIN_READ):
                return self._copies(namespace, records, where)

    def insert(
        self,
        namespace: str,
        records: Sequence[MemoryRecord],
        *,
        vectors: Mapping[str, Vector] = NO_VECTORS,
        dedupe: Filter | None = None,
    ) -> builtins.list[MemoryRecord]:
        rows = {record.id: _row(record) for record in records}
        # The vector of a record not written finds no row to go with.
        given = [(vector.model, vector.data, record_id) for record_id, vector in vectors.items()]
        with self._lock:
            ensure_open(self)
            if not records:
                return []
            with self._up_to_date(_BEGIN_WRITE):
                copies = None if dedupe is None else self._copies(namespace, records, dedupe)
                returned, written = split_batch(records, copies)
                reject_stored_keys(written, self._is_key_stored)
                check_sizes(vectors.values(), self._vector_size)
                self._connection.executemany(_INSERT, [rows[record.id] for record in written])
                self._connection.executemany(_ADD_WORDS, [(record.id,) for record in written])
                self._connection.executemany(_SET_VECTOR, given)
        return returned

    def modify(
        self, namespace: str, by: Lookup, value: str, build: Build, *, vector: Vector | None = None
    ) -> MemoryRecord:
        with self._lock:
            ensure_open(self)
            with self._up_to_date(_BEGIN_WRITE):
                current = self._find(namespace, by, value)
                record = build(current)
                after = vector_after(current, record, vector, self._vector_size)
                if current is None:
                    self._connection.execute(_INSERT, _row(record))
                    self._connection.execute(_ADD_WORDS, (record.id,))
                else:
                    self._connection.execute(_REPLACE, {**_row(record), "current_id": current.id})
                    if record.content != current.content:  # replaces the old text's words
                        self._connection.execute(_ADD_WORDS, (record.id,))
                self._give_vector(record, after)
        return record

    def delete(self, namespace: str, id: str) -> bool:
        with self._lock:
            ensure_open(self)
            with self._up_to_date(_BEGIN_WRITE):
                # while the row is there
                self._connection.execute(_DROP_WORDS, (namespace, id))
                self._connection.execute(_DROP_VECTOR, (namespace, id))
                deleted = self._connection.execute(_DELETE, (namespace, id)).rowcount
        return deleted > 0

    def purge_expired(self, now: datetime, *, limit: int) -> int:
        with self._lock:
            ensure_open(self)
            with self._up_to_date(_BEGIN_WRITE):
                rows = self._connection.execute(_EXPIRED_SEQS, (_write_time(now), limit))
                seqs = [seq for (seq,) in rows]
                given = json.dumps(seqs)
                for statement in _PURGE:
                    self._connection.execute(statement, (given,))
        return len(seqs)

    def list(
        self, namespace: str, *, where: Filter, limit: int | None, offset: int
    ) -> builtins.list[MemoryRecord]:
        conditions, values = _conditions(where)
        sql = f"{_SELECT}{conditions}{_PAGE}"  # noqa: S608 - fixed pieces only
        page = (namespace, *values, _NO_LIMIT if limit is None else limit, offset)
        with self._lock:
            ensure_open(self)
            with self._up_to_date(_BEGIN_READ):
                rows = self._connection.execute(sql, page).fetchall()
            return [_record(row) for row in rows]

    def count(self, namespace: str, *, where: Filter) -> int:
        conditions, values = _conditions(where)
        with self._lock:
            ensure_open(self)
            with self._up_to_date(_BEGIN_READ):
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

    def search_vectors(
        self, namespace: str, query: Vector, *, where: Filter, limit: int
    ) -> builtins.list[SearchHit]:
        conditions, values = _conditions(where)
        sql = f"{_SEARCH_VECTORS}{conditions}{_IN_ORDER}"  # noqa: S608 - fixed pieces only
        with self._lock:
            ensure_open(self)
            from pybodn import _vectors  # noqa: PLC0415 - NumPy, only once there is a model

            # The vectors must be stored as this process reads them.
            with self._up_to_date(_BEGIN_READ):
                check_size(query, self._vector_size(query.model))
                rows = self._connection.execute(sql, (namespace, query.model, *values))
                found = _vectors.best(query, ((seq, vector) for seq, vector in rows), limit=limit)
                records = self._records_at([seq for _, seq in found])
            return [SearchHit(record=records[seq], score=score) for score, seq in found]

    def last_seq(self) -> int:
        with self._lock:
            ensure_open(self)
            with self._up_to_date(_BEGIN_READ):
                last: int = _single(self._connection, _LAST_SEQ)
        return last

    def missing_vectors(
        self, model: str, *, after: int, until: int, limit: int
    ) -> builtins.list[MissingVector]:
        with self._lock:
            ensure_open(self)
            with self._up_to_date(_BEGIN_READ):
                rows = self._connection.execute(
                    _MISSING_VECTORS, (after, until, model, limit)
                ).fetchall()
        return [_missing(row) for row in rows]

    def write_vectors(self, vectors: Sequence[tuple[MissingVector, Vector]]) -> int:
        given = [
            (vector.model, vector.data, missing.id, vector.text) for missing, vector in vectors
        ]
        with self._lock:
            ensure_open(self)
            with self._up_to_date(_BEGIN_WRITE):
                check_sizes((vector for _, vector in vectors), self._vector_size)
                written: int = self._connection.executemany(_GIVE_VECTOR, given).rowcount
        return written

    def close(self) -> None:
        with self._lock:
            self._closed = True
            with _storage_errors():
                self._connection.close()  # closing again does nothing

    @contextmanager
    def _up_to_date(self, begin: str) -> Iterator[None]:
        """A transaction on a file still at the versions this process opened.

        A newer pybodn may have upgraded the file since. This process would then write the old
        way, leaving the file partly up to date, search an index built under other rules, or
        show records the newer one hides, so it stops instead.
        """
        with _storage_errors(), _transaction(self._connection, begin):
            if _file_versions(self._connection) != self._versions:
                raise StorageError(
                    "a newer pybodn has upgraded the file since this process opened it:"
                    " restart the process to use it"
                )
            yield

    def _find(self, namespace: str, by: Lookup, value: str) -> MemoryRecord | None:
        rows = self._connection.execute(_SELECT_BY[by], (namespace, value)).fetchall()
        return _record(rows[0]) if rows else None

    def _copies(
        self, namespace: str, records: Sequence[MemoryRecord], where: Filter
    ) -> dict[TextAndKey, MemoryRecord]:
        """``find_copies``, within a transaction.

        The query finds the first record of each text and key among those that hold a text of
        ``records`` and a key of them or none, so only the pairs asked for are kept here.
        """
        texts = [record.content for record in records]
        keys = [record.key for record in records if record.key is not None]
        as_bytes = any(_NUL in value for value in (*texts, *keys))
        compared = _AS_BYTES if as_bytes else _AS_WRITTEN
        conditions, values = _conditions(where)
        sql = f"{_COPIES}{compared}{conditions}{_FIRST_OF_EACH}"  # noqa: S608 - fixed pieces only
        given = (_json_of(texts, as_bytes=as_bytes), _json_of(keys, as_bytes=as_bytes))
        rows = self._connection.execute(sql, (namespace, *given, *values)).fetchall()
        wanted = {text_and_key(record) for record in records}
        found = (_record(row) for row in rows)
        return {text_and_key(record): record for record in found if text_and_key(record) in wanted}

    def _is_key_stored(self, namespace: str, key: str) -> bool:
        return bool(self._connection.execute(_KEY_USED, (namespace, key)).fetchall())

    def _vector_size(self, model: str) -> int | None:
        """How many bytes long the stored vectors of ``model`` are; ``None`` if there are none."""
        rows = self._connection.execute(_VECTOR_SIZE, (model,)).fetchall()
        return rows[0][0] if rows else None

    def _give_vector(self, record: MemoryRecord, vector: Vector | Unchanged | None) -> None:
        """Give the record just written ``vector``: a new one, the one it has, or none."""
        if isinstance(vector, Vector):
            self._connection.execute(_SET_VECTOR, (vector.model, vector.data, record.id))
        elif vector is None:
            self._connection.execute(_DROP_VECTOR, (record.namespace, record.id))

    def _records_at(self, seqs: Sequence[int]) -> dict[int, MemoryRecord]:
        """The records at these places in the order added, by place."""
        rows = self._connection.execute(_RECORDS_AT, (json.dumps(seqs),)).fetchall()
        return {row["seq"]: _record(row) for row in rows}


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
    """Refuse an SQLite that lacks what the file needs, before the file is changed."""
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
    if where.hide_archived:
        sql.append(_NOT_ARCHIVED)
    if where.hide_expired_at is not None:
        sql.append(_NOT_EXPIRED)
        values.append(_write_time(where.hide_expired_at))
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
        "expires_at": _write_time_or_none(record.expires_at),
        "archived_at": _write_time_or_none(record.archived_at),
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
            expires_at=_read_time_or_none(row["expires_at"]),
            archived_at=_read_time_or_none(row["archived_at"]),
        )
    except (TypeError, ValueError, RecursionError) as error:
        raise StorageError(f"record {row['id']!r} in the database is damaged: {error}") from error


def _missing(row: sqlite3.Row) -> MissingVector:
    """A row of ``_MISSING_VECTORS``. A text that is no longer text raises ``StorageError``."""
    if not isinstance(row["content"], str):
        raise StorageError(
            f"record {row['id']!r} in the database is damaged: its content is not text"
        )
    return MissingVector(
        seq=row["seq"], namespace=row["namespace"], id=row["id"], text=row["content"]
    )


def _json_of(texts: Sequence[str], *, as_bytes: bool) -> str:
    """``texts`` as a JSON list, each as written, or as SQLite's ``hex()`` shows its UTF-8
    bytes."""
    return json.dumps([text.encode().hex().upper() for text in texts] if as_bytes else texts)


def _json(value: object) -> str:
    # Keys stay in the order given, as the in-memory store keeps them.
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _write_time(moment: datetime) -> str:
    """UTC as text of one width, microseconds always written, so text order is time order."""
    return moment.astimezone(UTC).isoformat(timespec="microseconds")


def _write_time_or_none(moment: datetime | None) -> str | None:
    """``moment`` as ``_write_time`` writes it, or ``None`` for never."""
    return None if moment is None else _write_time(moment)


def _read_time_or_none(text: str | None) -> datetime | None:
    """A time ``_write_time_or_none`` wrote, or ``None`` for never."""
    return None if text is None else datetime.fromisoformat(text)
