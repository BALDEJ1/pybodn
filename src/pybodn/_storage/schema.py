"""The layout of a pybodn database file, and the steps that bring an older file up to it."""

# Marks a file as pybodn's, in the header where SQLite keeps an application id: "PYBD" in ASCII.
APPLICATION_ID = 0x50594244

# The word index: FTS5 over the words pybodn._words makes, keeping its own copy of them, and
# over a token for each record's namespace, so that a search reads only its namespace's rows.
# One without a copy would need SQLite 3.43, newer than Python's SQLite on many servers still
# in use. Migration 2 is made from it, so once released it never changes.
WORD_INDEX = "fts5(words, namespace_token, tokenize='ascii')"

# Migration ``n`` turns a file of version ``n - 1`` into version ``n``. Files already written
# have the layout these steps made, so a released step never changes: add a new one instead.
MIGRATIONS: tuple[tuple[str, ...], ...] = (
    (
        # seq is the order of insertion. It is a column of its own because the hidden row
        # number may change when the file is compacted; seq never does.
        """CREATE TABLE memories (
    seq           INTEGER PRIMARY KEY,
    id            TEXT NOT NULL UNIQUE,
    namespace     TEXT NOT NULL,
    memory_key    TEXT,
    content       TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    tags_json     TEXT NOT NULL,
    importance    REAL NOT NULL,
    source        TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    version       INTEGER NOT NULL
)""",
        "CREATE UNIQUE INDEX ux_memories_namespace_key ON memories (namespace, memory_key) "
        "WHERE memory_key IS NOT NULL",
        "CREATE INDEX ix_memories_namespace_seq ON memories (namespace, seq)",
    ),
    (
        # The words of each record, made by pybodn._words, and its namespace's token, under the
        # record's seq. FTS5's ascii tokenizer splits the words at the spaces between them and
        # nowhere else, and keeps the token whole.
        f"CREATE VIRTUAL TABLE memory_words USING {WORD_INDEX}",
        # Facts about the file itself: for now, the rules its word index was built with.
        "CREATE TABLE settings (name TEXT PRIMARY KEY, value NOT NULL)",
        "CREATE INDEX ix_memories_namespace_created ON memories (namespace, created_at)",
        # The vector of each record that has one, under the record's seq: the model that made it,
        # and its numbers at unit length, as pybodn._storage.base describes them. record_id keeps
        # a vector from the next record of its seq: SQLite gives the seq of a deleted last record
        # to the next one, and pybodn 0.1.0 deletes records without their vectors.
        """CREATE TABLE memory_vectors (
    seq       INTEGER PRIMARY KEY,
    record_id TEXT NOT NULL,
    model     TEXT NOT NULL,
    vector    BLOB NOT NULL
)""",
        # Finds how long a model's stored vectors are.
        "CREATE INDEX ix_memory_vectors_model ON memory_vectors (model)",
    ),
)
