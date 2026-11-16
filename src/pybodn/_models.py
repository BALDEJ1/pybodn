"""Memory records, the values they are created from, what a search finds, and the embedding
model that search by meaning uses."""

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field
from datetime import datetime
from typing import Literal

type JSONValue = JSONScalar | list[JSONValue] | dict[str, JSONValue]

# A JSON value that is not a list or an object: what a metadata filter compares a field with.
type JSONScalar = str | int | float | bool | None

# How a search looks for records: by words, by meaning, or by whatever the engine can.
type SearchMode = Literal["auto", "text", "vector"]


@dataclass(frozen=True, slots=True)
class NewMemory:
    """A memory to add. Its fields are validated when it is written."""

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
    """A stored memory. The engine creates it; ``version`` grows with every change.

    ``expires_at`` is when it expires, if ever, and ``archived_at`` when it was archived, if it
    is: reads leave such a record out unless asked.
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

    ``embed`` gets a list of at most ``batch_size`` texts and returns a vector for each, in the
    same order: a list of numbers, as long for every text. ``name`` names the model and its
    version: every vector is stored with it, so a model that gives other vectors needs another
    name.
    """

    name: str
    embed: Callable[[list[str]], Iterable[Iterable[float]]]
    _: KW_ONLY
    batch_size: int = 100
