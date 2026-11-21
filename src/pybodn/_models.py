"""Memory records, the values they are created from, what a search finds, and the embedding
model that search by meaning uses."""

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field
from datetime import datetime
from typing import Literal

# A value metadata can hold: what JSON can hold, with string keys.
type JSONValue = JSONScalar | list[JSONValue] | dict[str, JSONValue]

# A JSON value that is not a list or an object: what a metadata filter compares a field with.
type JSONScalar = str | int | float | bool | None

# How a search looks for records: by words, by meaning, or by whatever the engine can.
type SearchMode = Literal["auto", "text", "vector"]


@dataclass(frozen=True, slots=True)
class NewMemory:
    """A memory to add. Its fields are checked when it is written, and a field that fails its
    check raises ``ValidationError``.

    ``content`` is a non-empty string, and ``key``, if given, one too. ``metadata`` maps strings to
    JSON values, within the limits the README gives under What is checked. ``tags`` are non-empty
    strings without NUL characters, in a list or another iterable, never a single string; a repeated
    tag is kept once. ``importance`` is a number from 0 to 1. ``source``, any string, says where the
    memory came from. ``expires_at`` is a ``datetime`` with a time zone: from that moment on, the
    record is hidden.
    """

    content: str
    _: KW_ONLY
    key: str | None = None
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)
    tags: Sequence[str] = ()
    importance: float = 0.5
    source: str | None = None
    expires_at: datetime | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class MemoryRecord:
    """A stored memory. The engine creates it and gives it a new ``id``.

    The fields of ``NewMemory`` are as stored: ``tags`` as a tuple, and every moment in UTC.
    ``created_at`` and ``updated_at`` are when it was added and last changed, and ``version`` is
    1 when it is added and grows with every change. ``expires_at`` is when it expires, if ever,
    and ``archived_at`` when it was archived, if it is: reads leave such a record out unless
    asked.
    """

    id: str
    namespace: str
    content: str
    key: str | None
    metadata: dict[str, JSONValue]
    tags: tuple[str, ...]
    importance: float
    source: str | None
    created_at: datetime
    updated_at: datetime
    version: int
    expires_at: datetime | None
    archived_at: datetime | None


@dataclass(frozen=True, slots=True, kw_only=True)
class SearchHit:
    """A record a search found. A higher ``score`` is a better match, within one search only."""

    record: MemoryRecord
    score: float


@dataclass(frozen=True, slots=True)
class Embedder:
    """An embedding model, for search by meaning: ``embed`` turns texts into vectors.

    ``embed`` gets a list of texts, at most ``batch_size`` at a time, and returns a vector for
    each, in the same order: a list, tuple or array of numbers, as long for every text. An
    error it raises passes through unchanged, and an answer that cannot be used raises
    ``EmbeddingError``. ``batch_size`` is an integer from 1.
    ``name``, a non-empty string, names the model and its version: every vector is stored with
    it, so a model that gives other vectors needs another name.
    """

    name: str
    embed: Callable[[list[str]], Iterable[Iterable[float]]]
    _: KW_ONLY
    batch_size: int = 100
