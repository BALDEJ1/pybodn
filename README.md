# PyBodn

PyBodn is a Python library for keeping the memories of an application or an AI agent: short
texts such as facts about a user, notes, or what happened in a conversation. It stores them in an
SQLite file or in memory, and finds them again by words, or by meaning if you give it an
embedding model.

Each record belongs to a namespace, for example one per user, and can have a key, JSON metadata,
tags, an importance score and an expiry time. Several threads and processes can use the same
file, and there is an async version of the API.

PyBodn runs inside your program. It needs no server and does not call an LLM; the only model it
calls is the embedding model you pass in. The core uses only the standard library, and search by
meaning needs NumPy.

It is meant for a local agent, a desktop or command-line app, or a prototype. It is not meant for
a memory shared by several servers, because the SQLite file has to stay on one machine, or for
search by meaning over millions of records, because each search compares the query with every
vector in the namespace.

Code and files made with 1.0 keep working in every 1.x release, as Compatibility describes.

## Install

```bash
pip install pybodn
```

Search by meaning also needs NumPy. To install it with pybodn:

```bash
pip install "pybodn[vector]"
```

Requires Python 3.12 or newer.

## Quick start

```python
from pybodn import MemoryEngine

with MemoryEngine.sqlite("memory.db") as engine:
    notes = engine.collection("user:123")
    notes.add("Prefers short answers", tags=["style"])
    notes.put("language", "Python")  # creates the record with this key, or replaces it

    print([record.content for record in notes.list()])  # ['Prefers short answers', 'Python']
    print(notes.search("answer")[0].record.content)  # Prefers short answers
```

The records stay in `memory.db` after the program ends. Each example below runs on its own in an
empty folder, once the packages it names are installed.

To keep memories between runs of a program, open the same file each time:

```python
from pybodn import MemoryEngine


def remember(fact: str) -> None:
    with MemoryEngine.sqlite("agent.db") as engine:
        engine.collection("agent:assistant").add(fact, dedupe=True)


def recall(question: str) -> list[str]:
    with MemoryEngine.sqlite("agent.db") as engine:
        hits = engine.collection("agent:assistant").search(question, limit=3)
        return [hit.record.content for hit in hits]


remember("The user's name is Sam")  # one run of the agent
print(recall("What is my name?"))  # a later run, after a restart: ["The user's name is Sam"]
```

## Engines

An engine owns the storage. Create one with a factory below, never by calling the class, and use it
in `with`, which closes it at the end, or call `close()` yourself. Closing twice is safe; any other
call after closing raises `EngineClosedError`.

| Factory | Keeps records | Shared by |
|---|---|---|
| `MemoryEngine.in_memory(*, embedder=None)` | in this process, until `close()` | the threads of one process |
| `MemoryEngine.sqlite(path, *, timeout=5.0, embedder=None)` | in the SQLite file at `path`, created if missing | threads and processes |

Use `in_memory()` for tests and for data you do not need to keep. It searches by reading every
record, so keep it small. `embedder` is the embedding model that search by meaning uses.

`path` is a string or a `pathlib.Path`, and its folder must exist, or `StorageError` is raised.
`":memory:"` as the path keeps an SQLite database in memory, gone when the engine closes.

While another process writes to the file, a write waits up to `timeout` seconds, then raises
`StorageError`, and so does opening the file. With `timeout=0` they do not wait at all. A read
never waits for another process.

## Collections

`engine.collection(namespace)` gives you the records of one namespace, such as a user, an agent
or a conversation. Namespaces never see each other's records, and the same key in two
namespaces names two different records. A collection holds no records itself, so getting one is
cheap.

Namespaces do share the word counts that a search by words uses to rank records (see By words).
If clients must be kept fully apart, give each one its own file.

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
| `version` | `int` | `1` | grows by one with every change of the record |
| `expires_at` | `datetime \| None` | `None` | in UTC; from this moment on the record is hidden |
| `archived_at` | `datetime \| None` | `None` | in UTC; when `archive` hid the record, or `None` |

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
            NewMemory("Has two cats", tags=["profile"]),
            NewMemory("Works night shifts", importance=0.3),
        ]
    )
    print(notes.count())  # 3
```

### Skip copies: `dedupe=True`

`add(..., dedupe=True)` and `add_many(..., dedupe=True)` skip a memory when the namespace shows a
record with the same text and the same key, or both without a key, and return the first such
record in its place, as stored: the memory's other fields are not applied. Texts must match in
every character. Archived and expired records are not copies, though they still hold their
keys. In one `add_many`, a memory that repeats an earlier one gets that one's record. Two writers
adding the same text at once leave one record. With an embedding model, only new texts go to
it.

```python
from pybodn import MemoryEngine, NewMemory

with MemoryEngine.in_memory() as engine:
    notes = engine.collection("user:123")
    first = notes.add("Allergic to peanuts", tags=["health"])

    again = notes.add("Allergic to peanuts", tags=["food"], dedupe=True)
    print(again == first, again.tags)  # True ('health',)

    added = notes.add_many([NewMemory("Has two cats"), NewMemory("Has two cats")], dedupe=True)
    print(added[0] == added[1], notes.count())  # True 2
```

### Read: `get`, `get_by_key`, `list` and `count`

`get(id)` and `get_by_key(key)` return the record, or `None` if there is none or it is hidden.
`list()` returns every record in the order they were added; `put` and `update` keep a record in
its place.
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

### Filters

`list`, `count` and `search` take filters by name. A record is kept only if it passes every
filter given, and a filter set to `None` sets no condition. `limit` and `offset` page through
the records kept.

| Filter | Keeps a record that |
|---|---|
| `tags` | has every one of these tags |
| `metadata` | has each of these top-level fields, equal to its value as JSON: `True` is not `1`, and `None` is a field set to `None` |
| `min_importance` | has at least this importance |
| `created_since` | was created at this moment or after it, a `datetime` with a time zone |
| `created_before` | was created before this moment, a `datetime` with a time zone |

Hidden records are left out unless `include_archived=True` or `include_expired=True` asks for
them, as Hide below describes. `get` and `get_by_key` take these two as well.

```python
from datetime import UTC, datetime, timedelta

from pybodn import MemoryEngine

with MemoryEngine.in_memory() as engine:
    notes = engine.collection("user:123")
    notes.add("Allergic to peanuts", tags=["health"], importance=0.9)
    notes.add("Plays the guitar", metadata={"topic": "music"})

    print(notes.count(tags=["health"]))  # 1
    print(notes.list(metadata={"topic": "music"})[0].content)  # Plays the guitar
    print(notes.count(min_importance=0.8))  # 1
    print(notes.count(created_since=datetime.now(UTC) - timedelta(hours=1)))  # 2
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
calling it twice is safe. The key of a deleted record is free to use again. Its data stays in the
file until `engine.erase_deleted()` erases it, as Erase deleted data describes.

```python
from pybodn import MemoryEngine

with MemoryEngine.in_memory() as engine:
    notes = engine.collection("user:123")
    record = notes.put("language", "Python")

    print(notes.delete(record.id))  # True
    print(notes.delete(record.id))  # False
    notes.add("Prefers Rust", key="language")  # the key is free again
```

### Hide: `archive`, `restore` and `expires_at`

A hidden record stays stored, but reads leave it out. `archive(id)` hides a record until
`restore(id)` brings it back; both return the record and raise `RecordNotFoundError` if there is
none. `expires_at`, a `datetime` with a time zone that `add`, `add_many`, `put` and `update` take,
hides a record from that moment on, and `update(id, expires_at=None)` clears it.

- `get`, `get_by_key`, `list`, `count` and `search` leave hidden records out unless
  `include_archived=True` or `include_expired=True` asks for them.
- `update`, `put`, `delete`, `archive` and `restore` still reach them.
- A hidden record keeps its key: `add` with that key raises `DuplicateKeyError`, and `put`
  replaces the record with every field from the call, so it is no longer archived.
- `update` leaves a record archived or not as it was. `archive` and `restore` change `version`
  and `updated_at` only when they change the record.
- Hidden records still count in the scores of a search by words, and `embed_missing()` gives
  them vectors too.

`engine.purge_expired()` deletes every expired record, in every namespace, and returns how many.
It deletes a thousand at a time, each batch at once, and goes by the moment it began. Archived
records stay until you delete them.

```python
from datetime import UTC, datetime, timedelta

from pybodn import MemoryEngine

with MemoryEngine.in_memory() as engine:
    notes = engine.collection("user:123")
    old = notes.add("Uses Python 3.11")
    notes.archive(old.id)
    notes.add("Is on vacation this week", expires_at=datetime.now(UTC) + timedelta(days=7))
    notes.add("Was on a trip last week", expires_at=datetime.now(UTC) - timedelta(days=1))

    print([record.content for record in notes.list()])  # ['Is on vacation this week']
    print(notes.count(include_archived=True, include_expired=True))  # 3
    print(notes.get(old.id))  # None
    print(notes.restore(old.id).content)  # Uses Python 3.11
    print(engine.purge_expired())  # 1
```

### Erase deleted data: `erase_deleted`

A delete hides a record at once, but its data stays in the file until `engine.erase_deleted()`
erases it. So do the old values that `put` and `update` replace, and the records that
`purge_expired()` deletes. Until then, a program that reads the file itself, not through pybodn,
can still find them.

`erase_deleted()` rewrites the file and the log beside it without the data deleted, replaced or
purged before the call. Archived records, and expired records that `purge_expired()` has not deleted
yet, are still stored, so they stay: delete them first to erase them. Searches find the same records
afterwards.

- Run it now and then, not after every delete, at a time when nothing else writes. It rewrites the
  whole file, so its time grows with the size of the file, and other writers wait for it up to their
  `timeout`, then raise `StorageError`.
- It needs free disk space up to twice the size of the file.
- If another connection keeps the file busy for the whole `timeout`, by reading or writing, or the
  file cannot be written, it raises `StorageError`. The file is intact, but deleted data may still
  be in it, so call it again later.
- It erases the file and its log, and nothing else: copies and backups of the file, the disk
  under it, which may keep old blocks, and copies that your program or your embedding model's
  provider made are out of its reach. If you need more, keep the file on an encrypted disk.
- In memory, there is nothing to erase: a deleted record is gone at once.

```python
from pybodn import MemoryEngine

with MemoryEngine.sqlite("memory.db") as engine:
    notes = engine.collection("user:123")
    notes.delete(notes.add("Phone number is 555-0100").id)

    engine.erase_deleted()  # the number is gone from memory.db and its log
```

## Search

`search(query, *, limit=10, mode="auto", **filters)` returns the records that best match
`query`, best first: at most `limit` of them, as `SearchHit` objects that hold the `record` and
its `score`. A higher score is a better match, and scores compare only within one search. The
filters are those of `list`. A query without words, such as "?!", finds nothing.

| `mode` | Searches |
|---|---|
| `"auto"`, the default | by words, and with an embedding model by meaning too, mixing the two |
| `"text"` | by words |
| `"vector"` | by meaning, with the engine's embedding model |

### By words

A record matches if it holds any word of the query.

- Case, stress marks and apostrophes do not matter: "Алергі́я" finds "алергія", and
  "п’ять" finds "п'ять".
- A word also finds its forms with another ending, in Ukrainian, Russian and English: "алергії"
  finds "алергія", and "projects" finds "project". A longer word with no known ending finds the
  words that start with it: "лікар" finds "лікаря".
- A word of three letters or fewer, or a number, finds only itself: "про" does not find
  "проєкт", nor does "100" find "1000".
- Forms that change inside the word, such as "рік" and "року" or "run" and "running", are not
  found, and a short start may find another word: "this" finds "thing".
- Accents on Latin letters count: "cafe" does not find "café".
- Languages written without spaces between words, such as Chinese, Japanese and Thai, are not
  split into words: a run of their letters between spaces or punctuation counts as one word.
  Search by meaning has no such limit.

Records score by BM25, as full-text search engines rank them: more of the query's words, rarer
words and the exact forms rank a record higher. How rare a word is counts over every record in
the file, so a word common in other namespaces counts for less, but records of other namespaces
are never returned. Equal scores keep the order the records were added in. A query of more than
100 different words raises `ValidationError`.

```python
from pybodn import MemoryEngine

with MemoryEngine.in_memory() as engine:
    notes = engine.collection("user:123")
    notes.add("Алергія на арахіс", tags=["health"])
    notes.add("Working on two projects")

    print(notes.search("алергії")[0].record.content)  # Алергія на арахіс
    print(notes.search("project")[0].record.content)  # Working on two projects
    print(notes.search("арахіс", tags=["work"]))  # []
```

### By meaning

Search by meaning finds the records closest in meaning to the query, even with no word in
common. It needs NumPy, `pip install "pybodn[vector]"`, and an embedding model of your own,
which turns texts into vectors of numbers. Give the model to the engine as an `Embedder`, a
plain class to create, not to subclass:

`Embedder(name, embed, *, batch_size=100)`

- `embed` gets a list of at most `batch_size` texts and returns a vector for each, in the same
  order: a list, tuple or array of numbers, as long for every text.
- `name` names the model and its version. Every vector is stored with it, and a search compares
  only the vectors of the engine's model, so give a model that makes other vectors, even a new
  version of the same model, another name.

Every write of a new text asks the model for its vector before it takes any lock, and writes
nothing if the model fails; the model's error passes through as it raised it. A write that
keeps the text, such as an `update` of the tags, never calls the model.

`search(query, mode="vector")` returns the `limit` records whose vectors are closest to the
query's, however far, each scored by its similarity to the query, from -1 to 1. A record
without a vector of the engine's model is not found until `embed_missing()`, below, gives it
one.

```python
# pip install "pybodn[vector]"
from pybodn import Embedder, MemoryEngine

TOPICS = [{"coffee", "espresso", "latte"}, {"peanut", "peanuts", "allergic"}]


def embed(texts: list[str]) -> list[list[float]]:
    """A toy model for this example: 1, then how many words of each topic a text holds."""
    return [[1, *(len(set(text.lower().split()) & topic) for topic in TOPICS)] for text in texts]


with MemoryEngine.in_memory(embedder=Embedder("toy-1", embed)) as engine:
    notes = engine.collection("user:123")
    notes.add("Drinks an espresso every morning")
    notes.add("Allergic to peanuts")

    for hit in notes.search("coffee", mode="vector"):  # no record holds the word
        print(hit.record.content, round(hit.score, 2))
    # Drinks an espresso every morning 1.0
    # Allergic to peanuts 0.32

    for hit in notes.search("morning coffee"):  # by words and by meaning, mixed
        print(hit.record.content, round(hit.score, 3))
    # Drinks an espresso every morning 0.033
    # Allergic to peanuts 0.016
```

With a real model, `embed` calls it. A local model from sentence-transformers, for example:

```py
from sentence_transformers import SentenceTransformer

from pybodn import Embedder, MemoryEngine

model = SentenceTransformer("all-MiniLM-L6-v2")
engine = MemoryEngine.sqlite("memory.db", embedder=Embedder("all-MiniLM-L6-v2", model.encode))
```

A hosted model works the same way: call its API in `embed`, and set `batch_size` to the number
of texts it takes in one request.

`engine.embed_missing()` gives a vector of the engine's model to every record that has none, in
every namespace, and returns how many got one. Call it once you give a file its first model, or
a new one: the records an engine without the model wrote have no vector, and the vectors of
another model are replaced. The records go to the model `batch_size` at a time, and each batch
is written as soon as the model answers: if the model fails, the batches before stay written,
and the next call goes on with the rest.

### Mixed: the default with a model

With an embedding model, the default `mode="auto"` searches by words and by meaning, each on its
own, and mixes the two by reciprocal rank fusion: each search gives its best 50 records, or
`limit` if more, and a record scores `1 / (60 + place)` for each search that found it, added
up, so that a record found both ways rises. Scores are small, such as 0.03, and equal scores
keep the order of the search by words.

There is no threshold: with a model, a search returns up to `limit` records even when none is
close, as search by meaning always finds the closest. To keep only close ones, search with
`mode="vector"` and check the scores. If the model fails, the search raises its error: to
search by words anyway, catch it and search again with `mode="text"`.

## Speed and size

Measured on an Apple M4 with Python 3.13 and SQLite 3.53 by
[`benchmarks/bench.py`](https://github.com/BALDEJ1/pybodn/blob/main/benchmarks/bench.py), which
you can run on your own machine. The records are 5 to 25 words long, with words as frequent as
in real text, all in one namespace. Each search returns the best 10, and each time is the
fastest of five runs.

| Search | 10,000 records | 100,000 records |
|---|---|---|
| By words: a word that one record holds | 0.2 ms | 1.4 ms |
| By words: the five most common words | 8.6 ms | 84 ms |
| By meaning: vectors of 384 numbers | 12 ms | 117 ms |
| By meaning: vectors of 1,536 numbers | 30 ms | 298 ms |
| By meaning: vectors of 3,072 numbers | 50 ms | 467 ms |
| Mixed: 1,536 numbers and the five most common words | 39 ms | 380 ms |

- Search by words reads only the records that hold a word of the query, so a rare word is found
  at once and a common one takes longer.
- Search by meaning compares the query with every vector in the namespace, so its time grows
  with the number of records and the length of the vectors. It reads the vectors in batches, so
  its memory stays the same however many records there are: 26 MB with 1,536 numbers.
- Mixed search runs both and takes about as long as the two together. The time your model takes
  to turn the query into a vector comes on top.
- A search holds its engine's lock while it runs, so the engine's other calls wait for it; other
  engines and processes do not.
- A vector takes 4 bytes per number, so vectors make most of a file: 100,000 records with vectors
  of 1,536 numbers take 903 MB.
- The word index is one for the whole file, so a common word takes longer the more records the
  file holds: in a file of 1,000,000 records in 1,000 namespaces, a search in one namespace
  takes 0.1 ms for a word that one record holds, and 119 ms for the five most common words.
- `add` with `dedupe=True` looks for copies without an index, so it gets slower as the namespace
  grows. With an embedding model it takes 2.7 ms at 10,000 records and 27 ms at 100,000.
- In a namespace of 100,000 records where every other one has expired, `count()` takes 3.6 ms
  and the first page of `list()` 0.1 ms. `purge_expired()` deletes the 50,000 expired records in
  1.2 s, and `erase_deleted()` then takes 0.2 s.
- An async call takes about 30 µs longer than the same sync call, because it runs in a thread.

## Threads and processes

Any number of threads can share one engine. Processes share an SQLite file, each with an engine
of its own; an `in_memory()` engine is never shared between processes. After `os.fork()`, the
child process opens an engine of its own: an engine is never used on both sides of a fork.

- A call that fails writes nothing.
- Each call is a step of its own: two calls in a row are not one transaction, and another engine
  may write between them.
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

## Async

`AsyncMemoryEngine` and `AsyncCollection` are the engine and the collection for asyncio. Open an
engine with `await AsyncMemoryEngine.sqlite(path)` or `await AsyncMemoryEngine.in_memory()`, use
it in `async with`, and await every method but `collection()`. Each method takes the arguments of
the method of the same name above, and gives the same results and errors. It runs that method in
a thread, so the event loop runs on while a call waits for the file's lock or the embedding
model. Context variables reach the thread.

- Each event loop has four threads of its own for the calls of all its engines, so that calls
  waiting for the file do not take the threads the rest of the program shares. Further calls wait
  their turn.
- A call given up on, as on a timeout, once begun still runs to its end, so a write may still
  land.
- `close()` does not wait for a free thread.

```python
import asyncio

from pybodn import AsyncMemoryEngine


async def main() -> None:
    async with await AsyncMemoryEngine.sqlite("memory.db") as engine:
        notes = engine.collection("user:123")
        await asyncio.gather(notes.add("Prefers short answers"), notes.add("Has two cats"))

        print(await notes.count())  # 2
        print((await notes.search("answers"))[0].record.content)  # Prefers short answers


asyncio.run(main())
```

## Errors

Every error pybodn raises is in `pybodn.errors` and is a `PyBodnError`.

| Error | Raised when |
|---|---|
| `ValidationError` | an argument is invalid; the message names it. Also a `ValueError`. |
| `DuplicateKeyError` | `add` or `add_many` uses a key the namespace already has, even a hidden record's, or one batch repeats a key |
| `RecordNotFoundError` | `update`, `archive` or `restore` gets an id with no record. Also a `LookupError`. |
| `ConcurrentUpdateError` | `update` gets an `expected_version` that is not the record's version |
| `ConfigurationError` | the engine is not set up for what was asked: `mode="vector"` or `embed_missing()` without an embedding model, or a model without NumPy installed |
| `EmbeddingError` | the model's answer is unusable: not one vector of numbers for each text, a vector that cannot be scaled to unit length, such as all zeros, or one of another length than the model's stored vectors |
| `StorageError` | the storage fails: the file cannot be opened, a write waited longer than `timeout`, the file is not pybodn's, is in a newer file format or holds triggers or views, a rollback journal stays beside it, its SQLite lacks full-text search or JSON, a newer pybodn upgraded it after this process opened it, another program damaged a record in it, or `erase_deleted()` could not finish. The original error, if there is one, is its `__cause__`. |
| `EngineClosedError` | the engine is used after `close()` |

An unknown keyword argument raises `TypeError`, as it does for any Python function. An error
the embedding model raises passes through unchanged. A message may name the id, key or namespace
of a record and quote the start of an invalid value, so log messages with the same care as the
data.

## What is checked

Every value is checked before anything is written, the same way for both storages, so data that
fits in memory also fits in the file. Oversized or self-containing data fails at once with
`ValidationError` instead of slowing the call down. The length of `content`, keys, tags and
namespaces is not limited, and the limits below never get stricter in a 1.x release.

- `namespace`, `content`, keys, ids and tags are non-empty strings. Every string must be
  encodable as UTF-8, so a lone surrogate such as `"\ud800"` is rejected.
- `tags` is a list or another iterable of strings, never a single string.
- Tags, and every key and text in `metadata`, hold no NUL character, `"\x00"`: older SQLite,
  such as 3.34, cuts the JSON they are kept in at one, so filters would go wrong.
- `importance` is a number from 0 to 1 and is stored as a `float`. `True`, `False` and NaN are
  rejected.
- `metadata` is a mapping with string keys and JSON values: `str`, `int`, `float`, `bool`,
  `None`, lists and dicts. Tuples and non-string keys are rejected rather than converted, and so
  are NaN and infinity. It holds at most 100 levels, 1,000,000 values and 16 MB of text, and its
  integers fit in 64 bits, as SQLite stores them.
- A record keeps a copy of the `metadata` you pass: changing your dict later changes nothing.
- `expires_at` is `None` or a `datetime` with a time zone, within years 1 to 9999 in UTC, and
  `include_archived` and `include_expired` are `True`, `False` or `None`. `dedupe` is `True` or
  `False`.
- `limit` and `expected_version` are `None` or an integer from 1, and `offset` is an integer
  from 0. None of them can exceed 2**63 - 1, the largest integer SQLite stores.
- A search's `query` is a string, of at most 100 different words when it searches by words. Its
  `limit` is an integer from 1, never `None`, and its `mode` is `"auto"`, `"text"` or `"vector"`.
- The filters hold at most 100 tags and 100 metadata fields, and the metadata fields are
  matched against `str`, `int`, `float`, `bool` or `None`, never a list or a dict.
- An `Embedder` given to an engine has a non-empty name, a callable `embed` and a `batch_size`
  from 1.
- `path` is a non-empty string or path object without NUL characters, and `timeout` is a number
  of seconds from 0 to 2,147,483 (almost 25 days).

## The SQLite file

- While the file is open, SQLite keeps two more files beside it: `memory.db-wal` and
  `memory.db-shm`. They hold changes not yet moved into the main file, so never delete them.
  They go away when the last engine on the file closes.
- A change is in the file before the call returns, so a crash of your program does not lose it.
- Every process using the file must run on the same computer. The file's write-ahead log does
  not work on a network drive ([SQLite docs](https://sqlite.org/wal.html)).
- Besides the records, the file keeps the words of every record, for search by words, and the
  vector of each record that has one.
- Opening a file needs SQLite with full-text search (FTS5) and JSON, as Python's own builds and
  the common Linux distributions have. Without them, `StorageError` says what is missing and the
  file stays as it was.
- The file is marked as pybodn's and carries a schema version. Opening a file made by an older
  pybodn brings it up to date, as Upgrading from an older release describes. A file that is not
  pybodn's, such as another program's SQLite database, one in a newer file format, which only pybodn
  2.0 or later can write, or one holding triggers or views, which pybodn never makes, raises
  `StorageError` and stays exactly as it was.
- If a rollback journal (`-journal`) stays beside the file for `timeout` seconds, opening raises
  `StorageError` rather than let SQLite apply it to the file.
- A process that opened the file before a newer pybodn upgraded it raises `StorageError` on its
  next read or write, asking to be restarted, instead of using the file the old way.

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

To check the file, use SQLite's `PRAGMA integrity_check`: it reads the whole file and answers
`ok` if nothing is damaged. To compact the file and erase deleted data from it, call
`engine.erase_deleted()`, as Erase deleted data describes.

```python
import sqlite3
from contextlib import closing

from pybodn import MemoryEngine

with MemoryEngine.sqlite("memory.db") as engine:
    engine.collection("user:123").add("Prefers Python")

with closing(sqlite3.connect("memory.db", autocommit=True)) as connection:
    print(connection.execute("PRAGMA integrity_check").fetchone())  # ('ok',)
```

## Data and security

- pybodn does not encrypt the file. On most systems other users of the machine can read it and the
  files beside it, so keep it in a folder that only your program's user can open. Whoever can create
  files in that folder can also change or destroy the data through the files SQLite keeps there.
- A namespace keeps records apart but does not guard them: code that has the engine can read
  every namespace.
- With an embedding model, the content of each record that gets a vector goes to the model, and
  so does the query of a search by meaning or a mixed search. Keys, tags, metadata and sources
  never do.
- For how deleted data leaves the file, see Erase deleted data.
- To report a vulnerability, follow the
  [security policy](https://github.com/BALDEJ1/pybodn/blob/main/SECURITY.md) rather than opening
  an issue.

## Upgrading from an older release

A file from 0.3.0 is already in the format of 1.x, and pybodn opens it without changing it. pybodn
refuses a file that holds triggers or views, so drop any of your own before you upgrade.

A file from 0.2.0 or 0.1.0 is brought up to date the first time pybodn opens it.

1. Stop every process that uses the file, and copy the file, as The SQLite file shows, to have one
   to go back to. A process still running 0.2.0 raises `StorageError` on its next write or search,
   but its other reads would show records that pybodn now hides, and one running 0.1.0 would add
   records that search never finds.
2. Upgrade pybodn and start the processes again. The first to open the file brings it up to date
   while it holds the file's write lock, on the machine of Speed and size: 0.1 s per 100,000
   records for a file from 0.2.0, which rebuilds two indexes, and 1.1 s for one from 0.1.0, which
   also indexes the words of every record. The others wait for it up to their
   `timeout`, then raise `StorageError`, so open a large file once on its own first.
3. For a file from 0.1.0 with an embedding model, call `engine.embed_missing()` once, so that
   search by meaning finds the records written before.

## Compatibility

pybodn follows [semantic versioning](https://semver.org/). Until 2.0, a release may fix bugs and
add things, but it keeps what this list covers:

- The names that `pybodn`, `pybodn.errors` and `pybodn.indexes` export.
- The names, order and defaults of parameters, which of them are passed by name, and what each
  call returns.
- The error classes and the classes they inherit from. When a call has more than one problem,
  the first that applies is raised: a closed engine, then an invalid or unknown argument, then a
  record that does not exist, then an error of the embedding model, then a version that does not
  match.
- The behaviour this README and the docstrings describe.
- The file format. Every 1.x release reads and writes a file of any other 1.x release, older or
  newer. A minor release may add a table or a field that an older 1.x does not notice; any other
  change to the file waits for 2.0.
- The `vector` extra.

These are not covered and may change in any release:

- Names that start with an underscore, and the constructors of the engine and collection classes:
  create engines with `sqlite()` and `in_memory()`, and collections with `collection()`.
- The text of error messages and of `repr`.
- How scores are computed, including the numbers under Mixed. Only this is kept: within one
  search, a higher score is a better match.
- Timings, the number of threads, and the figures under Speed and size.
- The tables inside the file. Other tools can read them, but their layout may change, and
  writing to them is not supported.
- Opening a read-only file to read from it, which works today.
- Search by words for characters that a newer Unicode added. Which characters are letters comes
  from the running Python's Unicode data, so an older Python may split a word that holds such a
  character differently, and not find it.

Before anything covered is removed, it is deprecated: using it gives a `DeprecationWarning` that
names what to use instead, and it goes only in 2.0. A wrong result is a bug and gets fixed, but
a fix that could break programs that rely on the wrong result waits for a minor release with a
warning, or for 2.0. Only the latest release gets fixes.

A Python version stops being supported only after its end of life, in a minor release, with a
line in the changelog, and the same goes for NumPy versions in the `vector` extra. The oldest
SQLite tested is 3.34, and pybodn does not check the version.

## Not in this release

- Approximate vector indexes, such as FAISS, and external vector databases: search by meaning
  reads every vector in the namespace.
- "Or" and "not" in filters, filters on nested metadata fields and on the time of the last
  change, and an upper bound on importance.
- Phrase search and highlighting of matches.
- Ranking by importance or recency: the order depends only on how well a record matches.
- Treating "café" as "cafe", and splitting languages written without spaces into words.
- Several embedding models in one file at once.
- Deleting the expired records of one namespace only, or only those expired before a moment:
  `purge_expired()` deletes them all.
- Telling from the result of `dedupe=True` whether a record was added or found, `expected_version`
  for `archive` and `restore`, and a filter that keeps only archived records.
- An embedding model written as a coroutine: the async engine runs the plain `embed` in its
  thread.
- A minimum score for search.
- Storage backends of your own: the storage interface is private for now.

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

## Contributing

How to set up the project, run its checks and propose a change is in
[CONTRIBUTING.md](https://github.com/BALDEJ1/pybodn/blob/main/CONTRIBUTING.md).

## License

[MIT](https://github.com/BALDEJ1/pybodn/blob/main/LICENSE)
