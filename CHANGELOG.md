# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Hidden records. `Collection.archive(id)` hides a record and `Collection.restore(id)` brings it back; `expires_at`, a moment with a time zone that `add`, `add_many`, `put` and `update` take, hides a record from that moment on, and `update(id, expires_at=None)` clears it. A hidden record stays stored: `get`, `get_by_key`, `list`, `count` and `search` leave it out unless `include_archived=True` or `include_expired=True` asks for it, while `update`, `put`, `delete`, `archive` and `restore` still reach it, and it keeps its key, so `add` with that key raises `DuplicateKeyError` and `put` replaces it, bringing it back from the archive. `update` leaves a record archived or not as it was. Archiving or restoring changes the version and `updated_at` only when it changes the record, and a missing record raises `RecordNotFoundError`. Hidden records still count in the scores of a search by words, and `embed_missing` gives them vectors too.
- `MemoryEngine.purge_expired()` deletes every expired record, in every namespace, and returns how many. It deletes a thousand at a time, each batch at once, and goes by the moment it began.
- `add(..., dedupe=True)` and `add_many(..., dedupe=True)` skip a memory when the namespace shows a record with the same text and the same key, or both without one, and return the first such record added in its place, as stored: the memory's other fields are not applied. Texts must match in every character. Archived and expired records are no copies, but still hold their keys. In a batch, a memory that repeats an earlier one gets that one's record, and only the memories written must not share a key. Two writers adding the same text at once leave one record. With an embedding model, only new texts go to the model. Copies are looked for among the namespace's records, without an index: with an embedding model, an `add` takes about 3 ms in a namespace of 10,000 records and 27 ms in one of 100,000, on an Apple M4.
- `AsyncMemoryEngine` and `AsyncCollection`, for asyncio: open one with `await AsyncMemoryEngine.sqlite(path)` or `await AsyncMemoryEngine.in_memory()`, use it in `async with`, and await every method but `collection()`. Each method does what the method of the same name of `MemoryEngine` or `Collection` does, with the same parameters, checks, errors and results, in a thread, so the event loop runs on while a call waits for the file's lock or the embedding model. Context variables reach the thread. Each event loop has four threads of its own for the calls of all its engines, so that calls waiting for the file do not take the threads the rest of the program shares; the other calls wait their turn. A call given up on, as on a timeout, once begun still runs to its end in its thread, so a write may still land, and keeps the thread until then. `close()` does not wait for a free thread, and once begun it runs to its end. No new dependency.

### Changed

- The PyPI description now mentions SQLite and search by words or by meaning.
- The SQLite file moves to schema version 3: each record keeps when it expires and when it was archived, and the indexes a namespace's reads go through hold both, so that a read leaves hidden records out without reading their rows. The first open of an older file rebuilds those two indexes while it holds the file's write lock, about 0.1 seconds per 100,000 records on an Apple M4.
- A process that opened a file before a newer pybodn upgraded it now also raises `StorageError` on its next read, not only on its next write or search, since the newer pybodn may hide records it would show.
- Tags, and every key and text in `metadata` at any depth, must not hold a NUL character, `"\x00"`, whether written or given to a filter: `ValidationError` says so, for both storages. Older SQLite, such as 3.34, cuts the JSON they are kept in at that character, so a filter on them kept the wrong records there. Records already stored with one keep it.

## [0.2.0] - 2026-11-16

### Added

- Filters for `list` and `count`, passed by name: `tags` (has every tag), `metadata` (each top-level field is there and equals its value, as JSON compares them: `True` is not `1`, and `None` is a field set to `None`), `min_importance`, and `created_since` and `created_before` (moments with a time zone). A record is kept only if it passes every filter given, a filter set to `None` sets no condition, and `limit` and `offset` page through the records kept.
- `Collection.search(query, *, limit=10, mode="auto", **filters)` returns the records that best match the query, best first, as `SearchHit` objects: the `record` and its `score`, where higher is better and scores compare only within one search. The search by words finds the records that hold any word of the query. Case, stress marks and apostrophes do not matter, and a word also finds its forms with another ending in Ukrainian, Russian and English ("алергії" finds "алергія", "projects" finds "project"); a word of three letters or fewer, or a number, finds only itself. Records score by BM25 over the words of every record in the file, so a word common in other namespaces counts for less, but records of other namespaces are never returned. It takes the filters `list` takes. A query without words returns nothing, and one of more than 100 different words raises `ValidationError`. `mode="text"` searches by words, and so does `"auto"` without an embedding model; `mode="vector"` searches by meaning, below.
- Search by meaning, with an embedding model of your own: `Embedder(name, embed, *, batch_size=100)`, passed as `MemoryEngine.sqlite(path, embedder=...)` or `MemoryEngine.in_memory(embedder=...)`. `embed` gets a list of at most `batch_size` texts and returns a vector, a list of numbers, for each; `name` names the model and its version, and is stored with every vector. A write of a new text asks the model for its vector first, outside the file's lock, and writes nothing if the model fails; a write that keeps the text never calls it. `search(query, mode="vector")` returns the `limit` records whose vectors are closest to the query's, however far, each scored by its similarity to the query from -1 to 1. Without an embedding model it raises `ConfigurationError`. With one, `mode="auto"`, the default, searches both ways and mixes the two by reciprocal rank fusion: each search gives its best 50 records, or `limit` if more, and a record scores `1 / (60 + place)` for each search that found it, added up, so that one found both ways rises; equal scores keep the order of the search by words. If the model fails, the search raises its error. It needs NumPy: `pip install "pybodn[vector]"`; without a model, NumPy is never imported.
- `MemoryEngine.embed_missing()` gives a vector of the engine's model to every record without one, such as those an engine without the model wrote, replacing the vectors of another model, and returns how many got one: this is how to move to a new model.
- `SearchHit`, and `ConfigurationError` and `EmbeddingError` in `pybodn.errors`. `EmbeddingError` means the model's answer is unusable: not one vector of numbers per text, a vector that cannot be scaled to unit length, or one of another length than the model's stored vectors.

### Changed

- The SQLite file moves to schema version 2. It keeps an index of the words of every record, for search, the vector of each record that has one, and an index on creation time, so `created_since` and `created_before` read only the records in their range. The first open of a file from 0.1.0 indexes every record while it holds the file's write lock, about 1.2 seconds per 100,000 records on an Apple M4, and other processes wait for it within their `timeout`.
- Opening a file needs SQLite with full-text search (FTS5) and JSON, as Python's own builds and the common Linux distributions have; the oldest tested is SQLite 3.34. Without them, `StorageError` says what is missing and the file stays as it was.
- A process that opened a file before a newer pybodn upgraded it raises `StorageError` on its next write or search, asking to be restarted, instead of writing the old way or searching an index built under other rules. Processes still running 0.1.0 have no such check: stop them before upgrading, or the records they add afterwards will be missing from search.

## [0.1.0] - 2026-11-13

### Added

- `MemoryEngine.sqlite(path, *, timeout=5.0)`: an engine that keeps records in an SQLite file, shared safely by threads and processes. While another process writes, a write waits up to `timeout` seconds. A file that is not pybodn's, or comes from a newer pybodn, is refused and left unchanged.
- `MemoryEngine.in_memory()`: an engine that keeps records in memory. Use it in `with`, or call `close()`.
- `engine.collection(namespace)` returns a `Collection` with `add`, `add_many` (all or nothing), `put` (create or replace by key), `get`, `get_by_key`, `update` (optionally checked against `expected_version`), `delete`, `list` (in the order added, with `limit` and `offset`) and `count`.
- `MemoryRecord`, `NewMemory` and `JSONValue`, and the errors in `pybodn.errors`.
- Metadata is checked before it is stored: at most 100 levels deep, 1,000,000 values and 16 MB of text, with integers that fit in 64 bits. Anything larger raises `ValidationError` instead of slowing the call down.

### Changed

- `HashIndex` now lives in a private module. Import it as `from pybodn.indexes import HashIndex`; the `pybodn.indexes.hash` path from 0.0.1 no longer works.

## [0.0.1] - 2026-11-13

### Added

- `HashIndex`: an exact-match in-process index with `set`, `get`, `delete`, `clear`, `in` and `len`.
- Type information for downstream type checkers (`py.typed`).
