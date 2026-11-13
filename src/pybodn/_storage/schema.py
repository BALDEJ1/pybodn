"""The layout of a pybodn database file, and the steps that bring an older file up to it."""

# Marks a file as pybodn's, in the header where SQLite keeps an application id: "PYBD" in ASCII.
APPLICATION_ID = 0x50594244

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
)
