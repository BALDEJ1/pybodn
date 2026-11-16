"""A stored record for the unit tests, made without a store: change its fields with
``dataclasses.replace``."""

from datetime import UTC, datetime

from pybodn import MemoryRecord

_NOW = datetime(2026, 11, 15, 12, 0, tzinfo=UTC)

RECORD = MemoryRecord(
    id="a" * 32,
    namespace="user:123",
    content="кава",
    key=None,
    metadata={},
    tags=(),
    importance=0.5,
    source=None,
    created_at=_NOW,
    updated_at=_NOW,
    version=1,
    expires_at=None,
    archived_at=None,
)
