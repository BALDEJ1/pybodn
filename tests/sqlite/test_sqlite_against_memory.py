"""After any run of writes both stores answer every read alike, and the SQLite file keeps its words
and vectors in step with its records.

A fixed seed draws 500 writes of every kind, each run on both stores, some through an engine
with the embedding model and some through one without it, while the clock moves on.
"""

import random
import sqlite3
from collections.abc import Callable
from contextlib import closing
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any, NamedTuple

import pytest

from pybodn import Collection, Embedder, MemoryEngine, MemoryRecord, NewMemory, SearchHit
from pybodn._storage.base import Store
from pybodn._storage.memory import InMemoryStore
from pybodn._storage.sqlite import SQLiteStore, _indexed_words, _namespace_token
from pybodn.errors import PyBodnError
from tests.contract.conftest import Clock
from tests.fake_model import FakeModel, vector_of

np = pytest.importorskip("numpy")

SEED = 20261117
STEPS = 500
MODEL = "fake-1"
NAMESPACES = ("user:1", "user:2")
# The last word is not one the model counts.
WORDS = ("кава", "чай", "молоко", "арахіс", "алергія", "python", "зранку")
KEYS = (None, None, "a", "b", "c")
TAGS = ("health", "food", "work")
MISSING_ID = "0" * 32
# Each kind of write, and how often it comes up against the others.
KINDS = {
    "add": 6,
    "add-many": 2,
    "put": 2,
    "update": 5,
    "delete": 1,
    "archive": 1,
    "restore": 1,
    "purge-expired": 1,
    "embed-missing": 1,
}
WITH_MODEL = 0.7  # how often a write of a collection goes through the engine with the model

type IdOf = Callable[[str], str]  # the id the side gives the record of this id in memory


class Side(NamedTuple):
    """One store, through an engine with the model and one without it."""

    engine: MemoryEngine
    plain: MemoryEngine

    def notes(self, namespace: str, *, with_model: bool = True) -> Collection:
        return (self.engine if with_model else self.plain).collection(namespace)


type Write = Callable[[Side, IdOf], object]


def side_on(store: Store, clock: Clock) -> Side:
    model = Embedder(MODEL, FakeModel(), batch_size=3)
    return Side(MemoryEngine(store, clock=clock, embedder=model), MemoryEngine(store, clock=clock))


def text(rng: random.Random) -> str:
    return " ".join(rng.choices(WORDS, k=rng.randint(1, 2)))


def fields(rng: random.Random, clock: Clock) -> dict[str, Any]:
    drawn: dict[str, Any] = {}
    if rng.random() < 0.3:
        drawn["tags"] = rng.sample(TAGS, rng.randint(0, 2))
    if rng.random() < 0.3:
        drawn["metadata"] = {"topic": rng.choice(TAGS)}
    if rng.random() < 0.3:
        drawn["importance"] = rng.choice([0.0, 0.5, 1.0])
    if rng.random() < 0.3:
        later = clock.now + timedelta(minutes=rng.randint(-10, 60))
        drawn["expires_at"] = rng.choice([None, later])
    return drawn


def draw(rng: random.Random, clock: Clock, stored: Callable[[str], list[MemoryRecord]]) -> Write:
    """A write of a random kind, with random fields, on a record of the store in memory."""
    kind = rng.choices([*KINDS], weights=[*KINDS.values()])[0]
    namespace, with_model = rng.choice(NAMESPACES), rng.random() < WITH_MODEL
    records = stored(namespace)
    target = rng.choice(records).id if records and rng.random() < 0.9 else MISSING_ID
    content, key = text(rng), rng.choice(KEYS)
    drawn, dedupe = fields(rng, clock), rng.random() < 0.5
    changes = {"content": content, **drawn} if rng.random() < 0.6 or not drawn else drawn
    memories = [
        NewMemory(text(rng), key=rng.choice(KEYS), **fields(rng, clock))
        for _ in range(rng.randint(1, 3))
    ]

    def notes(side: Side) -> Collection:
        return side.notes(namespace, with_model=with_model)

    writes: dict[str, Write] = {
        "add": lambda side, _: notes(side).add(content, key=key, dedupe=dedupe, **drawn),
        "add-many": lambda side, _: notes(side).add_many(memories, dedupe=dedupe),
        "put": lambda side, _: notes(side).put(key or "a", content, **drawn),
        "update": lambda side, id_of: notes(side).update(id_of(target), **changes),
        "delete": lambda side, id_of: notes(side).delete(id_of(target)),
        "archive": lambda side, id_of: notes(side).archive(id_of(target)),
        "restore": lambda side, id_of: notes(side).restore(id_of(target)),
        "purge-expired": lambda side, _: side.engine.purge_expired(),
        "embed-missing": lambda side, _: side.engine.embed_missing(),
    }
    return writes[kind]


def attempt(write: Write, side: Side, id_of: IdOf) -> object:
    """What ``write`` returns, or the type of the error it raises."""
    try:
        return write(side, id_of)
    except PyBodnError as error:
        return type(error)


def learn(ids: dict[str, str], in_memory: object, in_file: object) -> None:
    """Pair the ids of the records both stores returned for the same write."""
    if isinstance(in_memory, MemoryRecord) and isinstance(in_file, MemoryRecord):
        ids.setdefault(in_memory.id, in_file.id)
    if isinstance(in_memory, list) and isinstance(in_file, list):
        for one, other in zip(in_memory, in_file, strict=True):
            learn(ids, one, other)


def as_in_file(ids: dict[str, str], value: object) -> object:
    """``value`` from the store in memory, with the ids the file gave the same records."""
    if isinstance(value, MemoryRecord):
        return replace(value, id=ids[value.id])
    if isinstance(value, list):
        return [as_in_file(ids, item) for item in value]
    return value


def assert_same_hits(
    ids: dict[str, str], in_memory: list[SearchHit], in_file: list[SearchHit]
) -> None:
    assert [as_in_file(ids, hit.record) for hit in in_memory] == [hit.record for hit in in_file]
    assert [hit.score for hit in in_file] == pytest.approx([hit.score for hit in in_memory])


def assert_same_reads(ids: dict[str, str], memory: Side, file: Side, query: str) -> None:
    for namespace in NAMESPACES:
        mine, theirs = memory.notes(namespace), file.notes(namespace)
        reads: list[Callable[[Collection], object]] = [
            lambda notes: notes.list(),
            lambda notes: notes.list(include_archived=True, include_expired=True),
            lambda notes: notes.list(tags=["health"], metadata={"topic": "food"}),
            lambda notes: notes.count(min_importance=0.5),
        ]
        for read in reads:
            assert as_in_file(ids, read(mine)) == read(theirs)
        for mode in ("text", "vector", "auto"):
            assert_same_hits(
                ids,
                mine.search(query, mode=mode, limit=1000),
                theirs.search(query, mode=mode, limit=1000),
            )


def assert_file_in_step(path: Path) -> None:
    """Each record has the words of its text and namespace, and a vector only of its text."""
    with closing(sqlite3.connect(path)) as connection:
        records = connection.execute(
            "SELECT seq, id, namespace, content FROM memories ORDER BY seq"
        ).fetchall()
        words = connection.execute(
            "SELECT rowid, words, namespace_token FROM memory_words ORDER BY rowid"
        ).fetchall()
        vectors = connection.execute(
            "SELECT seq, record_id, model, vector FROM memory_vectors ORDER BY seq"
        ).fetchall()
    assert words == [
        (seq, _indexed_words(content), _namespace_token(namespace))
        for seq, _, namespace, content in records
    ]
    by_seq = {seq: (record_id, content) for seq, record_id, _, content in records}
    for seq, record_id, model, data in vectors:
        assert seq in by_seq, f"a vector left of deleted record {record_id}"
        assert (record_id, model) == (by_seq[seq][0], MODEL)
        content = by_seq[seq][1]
        unit = np.array(vector_of(content)) / np.linalg.norm(vector_of(content))
        assert np.allclose(np.frombuffer(data, dtype="<f4"), unit)


def test_both_stores_agree_after_every_write(tmp_path: Path) -> None:
    rng = random.Random(SEED)  # noqa: S311 - the same writes every run, no secret
    clock = Clock()
    path = tmp_path / "memory.db"
    memory = side_on(InMemoryStore(), clock)
    file = side_on(SQLiteStore(str(path), timeout=5.0), clock)
    ids: dict[str, str] = {}

    def stored(namespace: str) -> list[MemoryRecord]:
        return memory.notes(namespace).list(include_archived=True, include_expired=True)

    with memory.engine, file.engine:
        for step in range(STEPS):
            # Arrange
            write = draw(rng, clock, stored)

            # Act
            in_memory = attempt(write, memory, lambda record_id: record_id)
            in_file = attempt(write, file, lambda record_id: ids.get(record_id, record_id))

            # Assert
            learn(ids, in_memory, in_file)
            assert as_in_file(ids, in_memory) == in_file, f"step {step}"
            assert_same_reads(ids, memory, file, text(rng))
            assert_file_in_step(path)
            clock.now += timedelta(minutes=rng.randint(0, 5))
