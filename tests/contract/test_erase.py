"""Erasing deleted data changes nothing that a read can see."""

import pytest

from pybodn import Collection, MemoryEngine, MemoryRecord, SearchHit
from tests.contract.conftest import START


def everything(notes: Collection) -> tuple[list[MemoryRecord], list[SearchHit]]:
    """Every record of ``notes``, hidden ones too, and what a search by words finds."""
    return (
        notes.list(include_archived=True, include_expired=True),
        notes.search("python", mode="text", include_archived=True, include_expired=True),
    )


@pytest.mark.parametrize("fast", [False, True], ids=["full", "fast"])
def test_erase_deleted_keeps_every_stored_record_and_finds_the_same(
    engine: MemoryEngine, notes: Collection, fast: bool
) -> None:
    # Arrange: stored records, hidden ones among them, beside deleted and replaced ones
    notes.add("prefers python for backend work", key="language", tags=["work"])
    notes.put("language", "prefers python and rust for backend work")
    notes.archive(notes.add("wrote python in lisbon").id)
    notes.add("python on a call until noon", expires_at=START)
    notes.delete(notes.add("drinks green tea with python").id)
    before = everything(notes)

    # Act: twice, since the first fast erase of a file is a full one
    engine.erase_deleted(fast=fast)
    engine.erase_deleted(fast=fast)

    # Assert
    assert everything(notes) == before
