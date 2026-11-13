# PyBodn

A foundational memory engine for intelligent systems.

> **Status: early alpha (0.0.x).** The API will change between releases. Not ready for production use.

PyBodn aims to be a small, dependable memory library for Python applications and AI agents:
structured records, fast in-process indexes, local persistence and pluggable retrieval, without
tying you to a specific LLM, embedding provider or database.

Today it ships one building block: `HashIndex`, an exact-match in-process index.

## Install

```bash
pip install pybodn
```

Requires Python 3.12+. No runtime dependencies.

## Example

```python
from pybodn.indexes import HashIndex

index: HashIndex[str, str] = HashIndex()
index.set("user:123:language", "record_001")

index.get("user:123:language")  # "record_001"
index.get("missing")  # None
"user:123:language" in index  # True
index.delete("user:123:language")  # True
index.delete("user:123:language")  # False, deleting twice is safe
```

Keys can be any hashable value, for example a `(namespace, key)` tuple.

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

[MIT](LICENSE)
