# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Filters for `list` and `count`, passed by name: `tags` (has every tag), `metadata` (each top-level field is there and equals its value, as JSON compares them: `True` is not `1`, and `None` is a field set to `None`), `min_importance`, and `created_since` and `created_before` (moments with a time zone). A record is kept only if it passes every filter given, and `limit` and `offset` page through the records kept.

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
