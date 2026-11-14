"""Memory records and the values they are created from."""

from collections.abc import Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field
from datetime import datetime

type JSONValue = str | int | float | bool | None | list[JSONValue] | dict[str, JSONValue]

# A JSON value that holds no other: what a metadata filter compares a field with.
type JSONScalar = str | int | float | bool | None


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


@dataclass(frozen=True, slots=True, kw_only=True)
class MemoryRecord:
    """A stored memory. The engine creates it; ``version`` grows with every change."""

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
