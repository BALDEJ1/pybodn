# PyBodn

PyBodn stores memories for Python applications and AI agents: texts with a namespace, an optional
key, JSON metadata, tags and an importance score. It keeps them in memory or in an SQLite file
that several threads and processes can share. It needs nothing outside the Python standard
library and does not tie you to an LLM, an embedding provider or a database server.

> **Status: alpha.** The API may still change between 0.x releases. There is no search yet: you
> read a record by its id or its key, or list a namespace in the order its records were added.

## Install

```bash
pip install pybodn
```

Requires Python 3.12 or newer.

## Quick start

```python
from pybodn import MemoryEngine

with MemoryEngine.sqlite("memory.db") as engine:
    notes = engine.collection("user:123")
    notes.add("Prefers short answers", tags=["style"])
    notes.put("language", "Python")  # creates the record with this key, or replaces it

    for record in notes.list():
        print(record.content)
# Prefers short answers
# Python
```

The records stay in `memory.db` after the program ends. Every example below runs on its own in an empty folder.

## Engines

An engine owns the storage. Create one with a factory and use it in `with`, which closes it at
the end, or call `close()` yourself. Closing twice is safe; any other call after closing raises
`EngineClosedError`.

| Factory | Keeps records | Shared by |
|---|---|---|
| `MemoryEngine.in_memory()` | in this process, until `close()` | the threads of one process |
| `MemoryEngine.sqlite(path, *, timeout=5.0)` | in the SQLite file at `path`, created if missing | threads and processes |

Use `in_memory()` for tests and for data you do not need to keep.

`path` is a string or a `pathlib.Path`, and its folder must exist, or `StorageError` is raised.
`":memory:"` as the path
keeps an SQLite database in memory, gone when the engine closes.

While another process writes to the file, a write waits up to `timeout` seconds, then raises
`StorageError`. With `timeout=0` it does not wait at all. A read never waits for another
process.

## Collections

`engine.collection(namespace)` gives you the records of one namespace, such as a user, an agent
or a conversation. Namespaces never see each other's records, and the same key in two
namespaces names two different records. A collection holds no records itself, so getting one is
cheap.

```python
from pybodn import MemoryEngine

with MemoryEngine.in_memory() as engine:
    alice = engine.collection("user:alice")
    bob = engine.collection("user:bob")

    note = alice.add("Allergic to peanuts")
    alice.put("language", "Python")
    bob.put("language", "Rust")

    print(bob.get(note.id))  # None
    print(bob.count())  # 1
```

## Records

Every call that writes returns a `MemoryRecord`. A record cannot be changed in place: `put` and
`update` return a new one. Changing the `metadata` dict of a record you hold changes only your
copy, never what is stored.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `id` | `str` | generated | 32 random hex characters |
| `namespace` | `str` | | the namespace of the collection |
| `content` | `str` | required | the memory itself; not empty |
| `key` | `str \| None` | `None` | a name for the record, unique within its namespace |
| `metadata` | `dict[str, JSONValue]` | `{}` | JSON values under string keys |
| `tags` | `tuple[str, ...]` | `()` | repeated tags are dropped, the first order is kept |
| `importance` | `float` | `0.5` | from 0 to 1 |
| `source` | `str \| None` | `None` | where the memory came from, as free text |
| `created_at` | `datetime` | now | in UTC; never changes |
| `updated_at` | `datetime` | now | in UTC; set again by every change |
| `version` | `int` | `1` | grows by one with every `put` or `update` of the record |

## Operations

### Add: `add` and `add_many`

`add(content, *, key=None, metadata={}, tags=(), importance=0.5, source=None)` adds one record.
A key the namespace already has raises `DuplicateKeyError`, and the record with that key stays
as it was.

`add_many(memories)` takes any iterable of `NewMemory`, such as a list or a generator. A
`NewMemory` has the same fields as `add`. It adds all of them or none: if one is invalid or its
key is taken, nothing is written. It returns the records in the order given.

```python
from pybodn import MemoryEngine, NewMemory
from pybodn.errors import DuplicateKeyError

with MemoryEngine.in_memory() as engine:
    notes = engine.collection("user:123")

    record = notes.add(
        "Prefers dark mode",
        key="theme",
        metadata={"app": "editor", "since": 2024},
        tags=["preference", "ui"],
        importance=0.8,
        source="settings page",
    )
    print(record.version)  # 1

    try:
        notes.add("Prefers light mode", key="theme")
    except DuplicateKeyError:
        print("the key is taken")  # and the record above is unchanged

    notes.add_many(
        [
            NewMemory("Lives in Kyiv", tags=["profile"]),
            NewMemory("Works night shifts", importance=0.3),
        ]
    )
    print(notes.count())  # 3
```

### Read: `get`, `get_by_key`, `list` and `count`

`get(id)` and `get_by_key(key)` return the record, or `None` if there is none. `list()` returns
every record in the order they were added; `put` and `update` keep a record in its place.
`list(limit=..., offset=...)` returns one page of that list, and a page past the end is empty.
`count()` returns the number of records.

```python
from pybodn import MemoryEngine, NewMemory

with MemoryEngine.in_memory() as engine:
    notes = engine.collection("user:123")
    first = notes.add("Prefers dark mode", key="theme")
    notes.add_many(NewMemory(f"Note {number}") for number in range(1, 6))

    print(notes.get(first.id) == first)  # True
    print(notes.get_by_key("missing"))  # None
    print([record.content for record in notes.list(limit=2, offset=1)])  # ['Note 1', 'Note 2']
    print(notes.count())  # 6
```

### Replace by key: `put`

`put(key, content, *, metadata={}, tags=(), importance=0.5, source=None)` creates the record
with this key, or replaces it. A replaced record takes every field from the call: a field you
leave out goes back to its default and does not keep its old value. The record keeps its id,
`created_at` and place in `list()`, and its `version` grows by one. Use `put` for facts that
have one current value, such as a setting.

`put` always writes; it has no `expected_version`. To change a keyed record only if nobody
changed it since you read it, read it with `get_by_key` and call `update` with
`expected_version`, as shown under Threads and processes.

```python
from pybodn import MemoryEngine

with MemoryEngine.in_memory() as engine:
    notes = engine.collection("user:123")

    first = notes.put("language", "Python", tags=["preference"])
    second = notes.put("language", "Rust")

    print(second.id == first.id)  # True
    print(second.version)  # 2
    print(second.tags)  # () because tags were not passed
```

### Change some fields: `update`

`update(id, *, expected_version=None, content=..., metadata=..., tags=..., importance=...,
source=...)` changes only the fields you pass, and at least one is required. `source=None`
clears the source; `None` for any other field raises `ValidationError`. The key cannot be
changed: delete the record and `put` it under the new key. An id with no record raises
`RecordNotFoundError`. `expected_version` is explained under Threads and processes.

```python
from pybodn import MemoryEngine

with MemoryEngine.in_memory() as engine:
    notes = engine.collection("user:123")
    record = notes.add("Prefers Python", tags=["preference"], source="chat")

    changed = notes.update(record.id, content="Prefers Rust", source=None)
    print(changed.content, changed.tags, changed.source)  # Prefers Rust ('preference',) None
```

### Delete: `delete`

`delete(id)` removes the record and returns `True`, or returns `False` if there was none, so
calling it twice is safe. The key of a deleted record is free to use again.

```python
from pybodn import MemoryEngine

with MemoryEngine.in_memory() as engine:
    notes = engine.collection("user:123")
    record = notes.put("language", "Python")

    print(notes.delete(record.id))  # True
    print(notes.delete(record.id))  # False
    notes.add("Prefers Rust", key="language")  # the key is free again
```

## Threads and processes

Any number of threads can share one engine. Processes share an SQLite file, each with an engine
of its own; an `in_memory()` engine is never shared between processes.

- A call that fails writes nothing.
- A change is seen by every engine on the file as soon as the call returns.
- `put` and `update` read a record and write it back in one step, so changes to one record made
  at the same time are applied one after another, never mixed. `version` counts every one.
- `list()` gives the same order in every engine, and after a restart.

To change a record only if nobody changed it since you read it, pass the version you read as
`expected_version` to `update`. If the record has moved on, `update` raises
`ConcurrentUpdateError` and changes nothing.

```python
from pybodn import MemoryEngine
from pybodn.errors import ConcurrentUpdateError

with MemoryEngine.in_memory() as engine:
    notes = engine.collection("user:123")
    record = notes.add("Prefers Python")
    notes.update(record.id, content="Prefers Rust")  # another change gets there first

    try:
        notes.update(record.id, content="Prefers Go", expected_version=record.version)
    except ConcurrentUpdateError:
        latest = notes.get(record.id)  # read the record again and decide what to do
```

## Errors

Every error pybodn raises is in `pybodn.errors` and is a `PyBodnError`.

| Error | Raised when |
|---|---|
| `ValidationError` | an argument is invalid; the message names it. Also a `ValueError`. |
| `DuplicateKeyError` | `add` or `add_many` uses a key the namespace already has, or one batch repeats a key |
| `RecordNotFoundError` | `update` gets an id with no record. Also a `LookupError`. |
| `ConcurrentUpdateError` | `update` gets an `expected_version` that is not the record's version |
| `StorageError` | the storage fails: the file cannot be opened, a write waited longer than `timeout`, the file is not pybodn's or comes from a newer pybodn, or another program damaged a record in it. The original error is its `__cause__`. |
| `EngineClosedError` | the engine is used after `close()` |

An unknown keyword argument raises `TypeError`, as it does for any Python function.

## What is checked

Every value is checked before anything is written, the same way for both storages, so data that
fits in memory also fits in the file. Oversized or self-containing data fails at once with
`ValidationError` instead of slowing the call down.

- `namespace`, `content`, keys, ids and tags are non-empty strings. Every string must be
  encodable as UTF-8, so a lone surrogate such as `"\ud800"` is rejected.
- `tags` is a list or another iterable of strings, never a single string.
- `importance` is a number from 0 to 1 and is stored as a `float`. `True`, `False` and NaN are
  rejected.
- `metadata` is a mapping with string keys and JSON values: `str`, `int`, `float`, `bool`,
  `None`, lists and dicts. Tuples and non-string keys are rejected rather than converted, and so
  are NaN and infinity. It holds at most 100 levels, 1,000,000 values and 16 MB of text, and its
  integers fit in 64 bits, as SQLite stores them.
- A record keeps a copy of the `metadata` you pass: changing your dict later changes nothing.
- `limit` and `expected_version` are `None` or an integer from 1, and `offset` is an integer
  from 0. None of them can exceed 2**63 - 1, the largest integer SQLite stores.
- `path` is a non-empty string or path object without NUL characters, and `timeout` is a number
  of seconds from 0 to 2,147,483 (almost 25 days).

## The SQLite file

- While the file is open, SQLite keeps two more files beside it: `memory.db-wal` and
  `memory.db-shm`. They hold changes not yet moved into the main file, so never delete them.
  They go away when the last engine on the file closes.
- A change is in the file before the call returns, so a crash of your program does not lose it.
- Every process using the file must run on the same computer. The file's write-ahead log does
  not work on a network drive ([SQLite docs](https://sqlite.org/wal.html)).
- The file is marked as pybodn's and carries a schema version. Opening a file made by an older
  pybodn brings it up to date. A file that is not pybodn's, such as another program's SQLite
  database, or one made by a newer pybodn, raises `StorageError` and stays exactly as it was.

To copy the file, close every engine that uses it, or use SQLite's backup, which is safe while
engines are writing:

```python
import sqlite3
from contextlib import closing

from pybodn import MemoryEngine

with MemoryEngine.sqlite("memory.db") as engine:
    engine.collection("user:123").add("Prefers Python")

    with (
        closing(sqlite3.connect("memory.db")) as source,
        closing(sqlite3.connect("backup.db")) as target,
    ):
        source.backup(target)

with MemoryEngine.sqlite("backup.db") as copy:
    print(copy.collection("user:123").count())  # 1
```

## Not in this release

- Search and filters by text, metadata or tags.
- Expiry, archiving and clean-up of old records.
- Storage backends of your own: the storage interface is private for now.
- An async API.

## HashIndex

`HashIndex` is a dictionary-backed index from any hashable key to a value, for lookups inside
one process.

```python
from pybodn.indexes import HashIndex

index: HashIndex[str, str] = HashIndex()
index.set("user:123:language", "record_001")

print(index.get("user:123:language"))  # record_001
print(index.get("missing"))  # None
print("user:123:language" in index)  # True
print(index.delete("user:123:language"))  # True
print(index.delete("user:123:language"))  # False, so deleting twice is safe
```

It also has `clear()` and `len(index)`. `get` returns `None` both for a missing key and for a
stored `None`; use `in` to tell them apart. The index has no lock of its own: if threads share
it, guard it with one.

## Development

```bash
uv sync
uv run prek install    # secret and workflow checks before each commit
uv run ruff check
uv run ruff format --check
uv run mypy
uv run coverage run -m pytest
uv run coverage report
```

## License

[MIT](https://github.com/BALDEJ1/pybodn/blob/main/LICENSE)
