"""Write the file that the installed pybodn makes from a few fixed records, for the tests of files
from older releases.

Run it once for each release, from the repository root, with that release from PyPI:

    uv run --isolated --no-project --with pybodn==0.1.0 python -m tests.golden.make_golden
    uv run --isolated --no-project --with "pybodn[vector]==0.2.0" python -m tests.golden.make_golden
    uv run --isolated --no-project --with "pybodn[vector]==0.3.0" python -m tests.golden.make_golden
    uv run --isolated --no-project --with "pybodn[vector]==1.0.1" python -m tests.golden.make_golden

Each run writes tests/golden/pybodn-<release>.db, with only what that release can store. It
refuses to replace a file that is there: the tests rely on these files never changing.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

import pybodn
from pybodn import Collection, MemoryEngine

FOLDER = Path(__file__).parent
NAMESPACE = "user:1"
MODEL = "golden-topics"


class Record(NamedTuple):
    key: str | None
    content: str
    tags: list[str]
    metadata: dict[str, str]
    importance: float
    source: str | None


# What every release can store, in the order the records are added. The second has letters
# outside ASCII in every field but its content. Contents stay ASCII, so that every Python splits
# their words alike. The third has only defaults.
RECORDS = (
    Record("language", "prefers python for backend work", ["work"], {"topic": "code"}, 0.8, "chat"),
    Record("пес", "has a dog named rex", ["дім"], {"тема": "тварини"}, 0.5, "розмова"),
    Record(None, "drinks green tea every morning", [], {}, 0.5, None),
)
# One word of each record, in the same order. Each is also one number of a vector.
TOPICS = ("python", "dog", "tea")
# Before the last record goes in, one is added and deleted, so the last one takes its place in
# the table. The last one is then added with other words and updated to its own. Both hold a topic
# word, so a search shows them if their words or vectors were left behind.
DELETED = "had a dog named max"
FIRST_WORDS = "used to write python at night"
# A record in a namespace of letters outside ASCII, with a topic word of the other namespace.
OTHER_NAMESPACE = "команда:7"
OTHER = "orders tea for the office"
# From 0.3.0 on: a record that expired long ago, and one that is archived.
EXPIRED = "on a call until noon"
ARCHIVED = "lived in lisbon before"
LONG_AGO = datetime(2000, 1, 1, tzinfo=UTC)


def embed(texts: list[str]) -> list[list[float]]:
    """A one for each topic word the text holds, then a small constant, so no vector is zero."""
    return [[float(topic in text.split()) for topic in TOPICS] + [0.1] for text in texts]


def path_of(release: str) -> Path:
    return FOLDER / f"pybodn-{release}.db"


def add(notes: Collection, record: Record) -> str:
    """Add ``record`` and return its id."""
    return notes.add(
        record.content,
        key=record.key,
        tags=record.tags,
        metadata=record.metadata,
        importance=record.importance,
        source=record.source,
    ).id


def fill(engine: MemoryEngine) -> None:
    notes = engine.collection(NAMESPACE)
    *first, last = RECORDS
    for record in first:
        add(notes, record)
    notes.delete(notes.add(DELETED).id)
    last_id = add(notes, last._replace(content=FIRST_WORDS))
    notes.update(last_id, content=last.content)
    engine.collection(OTHER_NAMESPACE).add(OTHER)
    if hasattr(notes, "archive"):
        notes.add(EXPIRED, expires_at=LONG_AGO)
        notes.archive(notes.add(ARCHIVED).id)


def main() -> None:
    path = path_of(pybodn.__version__)
    if path.exists():
        raise SystemExit(f"{path} is there already: delete it by hand to write it again")
    if hasattr(pybodn, "Embedder"):
        engine = MemoryEngine.sqlite(path, embedder=pybodn.Embedder(MODEL, embed))
    else:
        engine = MemoryEngine.sqlite(path)
    with engine:
        fill(engine)
    leftovers = sorted(found.name for found in FOLDER.glob(f"{path.name}-*"))
    if leftovers:
        raise SystemExit(f"the file was not closed cleanly: {leftovers} are left beside it")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
