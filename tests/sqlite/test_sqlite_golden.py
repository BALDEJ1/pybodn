"""Files that released versions of pybodn wrote themselves: each still opens, keeps its records and
finds them. They were made once by tests/golden/make_golden.py and never change."""

import shutil
from pathlib import Path

import pytest

from pybodn import Embedder, MemoryEngine, MemoryRecord
from tests.golden.make_golden import (
    ARCHIVED,
    EXPIRED,
    LONG_AGO,
    MODEL,
    NAMESPACE,
    OTHER,
    OTHER_NAMESPACE,
    RECORDS,
    TOPICS,
    embed,
    path_of,
)
from tests.sqlite.support import run_sql

RELEASES = ("0.1.0", "0.2.0", "0.3.0")
WITH_VECTORS = ("0.2.0", "0.3.0")
# The release whose files 1.x writes: the file format is frozen until 2.0.
CURRENT_FORMAT = "0.3.0"
# Application id, schema version and word rules of that format.
FORMAT_MARKS = [(0x50594244, 3, 1)]


def fields_of(record: MemoryRecord) -> tuple[object, ...]:
    """The record in the shape of ``RECORDS``."""
    return (
        record.key,
        record.content,
        list(record.tags),
        record.metadata,
        record.importance,
        record.source,
    )


def format_marks(path: Path) -> list[tuple[object, ...]]:
    return run_sql(
        path,
        "SELECT application_id, user_version, value FROM pragma_application_id,"
        " pragma_user_version, settings WHERE name = 'word_rules'",
    )


@pytest.mark.parametrize("release", RELEASES)
def test_a_file_from_a_release_keeps_its_records_and_finds_them(path: Path, release: str) -> None:
    # Arrange
    shutil.copyfile(path_of(release), path)
    dog = RECORDS[1]

    # Act
    with MemoryEngine.sqlite(path) as engine:
        notes = engine.collection(NAMESPACE)
        listed = [fields_of(record) for record in notes.list()]
        found = [[hit.record.content for hit in notes.search(t, mode="text")] for t in TOPICS]
        filtered = notes.list(tags=dog.tags, metadata=dog.metadata)
        other = engine.collection(OTHER_NAMESPACE).search(TOPICS[2], mode="text")

    # Assert
    assert listed == list(RECORDS)
    assert found == [[record.content] for record in RECORDS]
    assert [record.content for record in filtered] == [dog.content]
    assert [hit.record.content for hit in other] == [OTHER]
    assert format_marks(path) == FORMAT_MARKS


@pytest.mark.parametrize("release", WITH_VECTORS)
def test_a_file_from_a_release_finds_records_by_their_stored_vectors(
    path: Path, release: str
) -> None:
    pytest.importorskip("numpy")  # search by meaning computes with NumPy
    # Arrange
    shutil.copyfile(path_of(release), path)

    # Act
    with MemoryEngine.sqlite(path, embedder=Embedder(MODEL, embed)) as engine:
        notes = engine.collection(NAMESPACE)
        best = [notes.search(topic, mode="vector", limit=1)[0] for topic in TOPICS]

    # Assert: each topic's own record, at the similarity its vector was written with
    assert [hit.record.content for hit in best] == [record.content for record in RECORDS]
    assert [hit.score for hit in best] == pytest.approx([1.0] * len(TOPICS))


def test_a_file_from_0_3_0_keeps_its_expired_and_archived_records(path: Path) -> None:
    # Arrange
    shutil.copyfile(path_of("0.3.0"), path)

    # Act
    with MemoryEngine.sqlite(path) as engine:
        notes = engine.collection(NAMESPACE)
        everything = notes.list(include_archived=True, include_expired=True)

    # Assert: hidden from ordinary reads, as the first test shows, but still there
    hidden = {record.content: record for record in everything}
    assert hidden[EXPIRED].expires_at == LONG_AGO
    assert hidden[ARCHIVED].archived_at is not None


def test_a_file_of_the_current_format_is_left_exactly_as_it_was(path: Path) -> None:
    # Arrange
    shutil.copyfile(path_of(CURRENT_FORMAT), path)

    # Act
    MemoryEngine.sqlite(path).close()

    # Assert
    assert path.read_bytes() == path_of(CURRENT_FORMAT).read_bytes()
    assert sorted(entry.name for entry in path.parent.iterdir()) == [path.name]
