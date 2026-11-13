# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `MemoryEngine.in_memory()`: an engine that keeps records in memory. Use it in `with`, or call `close()`.
- `engine.collection(namespace)` returns a `Collection` with `add`, `add_many` (all or nothing), `put` (create or replace by key), `get`, `get_by_key`, `update` (optionally checked against `expected_version`), `delete`, `list` (in the order added, with `limit` and `offset`) and `count`.
- `MemoryRecord`, `NewMemory` and `JSONValue`, and the errors in `pybodn.errors`.

### Changed

- `HashIndex` now lives in a private module. Import it as `from pybodn.indexes import HashIndex`; the `pybodn.indexes.hash` path from 0.0.1 no longer works.

## [0.0.1] - 2026-11-13

### Added

- `HashIndex`: an exact-match in-process index with `set`, `get`, `delete`, `clear`, `in` and `len`.
- Type information for downstream type checkers (`py.typed`).
