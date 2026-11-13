# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- `HashIndex` now lives in a private module. Import it as `from pybodn.indexes import HashIndex`; the `pybodn.indexes.hash` path from 0.0.1 no longer works.

## [0.0.1] - 2026-11-13

### Added

- `HashIndex`: an exact-match in-process index with `set`, `get`, `delete`, `clear`, `in` and `len`.
- Type information for downstream type checkers (`py.typed`).
