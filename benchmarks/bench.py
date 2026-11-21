"""Measures every speed and size the README quotes, on this machine.

Run it from the repository root, with NumPy installed, as ``uv sync`` installs it:

    uv run python -m benchmarks.bench

It takes a few minutes and writes up to 2 GB of files to a temporary folder, removed at the end.
Records are 5 to 25 words drawn from 50,000 made-up words, the n-th most common used 1/n as often
as the most common one, as words are in real text. Vectors are random: a search by meaning reads
every vector whatever its numbers, so its time does not depend on them.
"""

import asyncio
import platform
import random
import sqlite3
import tempfile
import time
import tracemalloc
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from itertools import accumulate, batched
from pathlib import Path
from typing import NamedTuple

import numpy as np

from pybodn import (
    AsyncCollection,
    AsyncMemoryEngine,
    Collection,
    Embedder,
    MemoryEngine,
    NewMemory,
    __version__,
)
from tests.sqlite.support import make_old_file

RECORDS = (10_000, 100_000)
DIMENSIONS = (384, 1_536, 3_072)
MAIN_DIMENSIONS = 1_536  # the vectors of the mixed search, the memory and the file size quoted
NAMESPACES = 1_000  # in the file of many namespaces, each with
PER_NAMESPACE = 1_000
UPGRADED = 100_000  # records in each file from an older pybodn
OLDER = {"0.1.0": 1, "0.2.0": 2}  # each older pybodn, with the schema version of its files
HIDDEN = 100_000  # records in the file where every other one has expired
EXPIRED_AT = datetime(2026, 1, 1, tzinfo=UTC)
ASYNC_CALLS = 1_000  # calls in each run that measures what awaiting a call adds
VOCABULARY = 50_000
WORDS_PER_RECORD = (5, 25)
BATCH = 1_000  # records added at a time
RUNS = 5  # each search runs this many times, and the fastest counts
LIMIT = 10  # records each search returns
RARE = "2026"  # a word one record holds: a number, so it finds only itself
COMMON_WORDS = 5


class Corpus:
    """Made-up texts, the same on every run."""

    def __init__(self) -> None:
        pick = random.Random(0)  # noqa: S311 - made-up texts, not secrets
        syllables = [consonant + vowel for consonant in "бвдзклмнпрст" for vowel in "аеиоуі"]
        words: dict[str, None] = {}  # not a set, whose order changes from run to run
        while len(words) < VOCABULARY:
            words["".join(pick.choices(syllables, k=pick.randint(2, 4)))] = None
        self.words = list(words)
        self._weights = list(accumulate(1 / rank for rank in range(1, VOCABULARY + 1)))

    @property
    def common(self) -> str:
        """A query of the most common words."""
        return " ".join(self.words[:COMMON_WORDS])

    def texts(self, count: int, *, seed: int = 0) -> Iterator[str]:
        """``count`` texts, the same for the same ``seed``."""
        pick = random.Random(seed)  # noqa: S311 - made-up texts, not secrets
        for _ in range(count):
            size = pick.randint(*WORDS_PER_RECORD)
            yield " ".join(pick.choices(self.words, cum_weights=self._weights, k=size))


class Measured(NamedTuple):
    """What a file of one namespace measures: times in seconds, sizes in bytes."""

    rare: float
    common: float
    meaning: float
    mixed: float
    dedupe: float  # an add with dedupe=True of a text the namespace does not hold
    memory: int
    file_size: int


def random_model(dimensions: int) -> Embedder:
    """A model that gives each text a vector of ``dimensions`` random numbers."""
    generator = np.random.default_rng(0)
    return Embedder(
        f"random-{dimensions}", lambda texts: generator.standard_normal((len(texts), dimensions))
    )


def fill(notes: Collection, texts: Iterator[str]) -> None:
    for batch in batched(texts, BATCH):
        notes.add_many(NewMemory(text) for text in batch)


def filled(engine: MemoryEngine, corpus: Corpus, namespace: str, records: int) -> Collection:
    """The collection of ``namespace``, given ``records`` records: first one that holds ``RARE``,
    then made-up texts."""
    notes = engine.collection(namespace)
    notes.add(RARE)
    fill(notes, corpus.texts(records - 1))
    return notes


def fastest(action: Callable[[], object]) -> float:
    """The shortest of ``RUNS`` runs of ``action``, in seconds: other programs on the machine
    only ever slow a run down, as Python's timeit documentation explains."""
    timings = []
    for _ in range(RUNS):
        started = time.perf_counter()
        action()
        timings.append(time.perf_counter() - started)
    return min(timings)


def peak_memory(search: Callable[[], object]) -> int:
    tracemalloc.start()
    try:
        search()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def one_namespace(folder: Path, corpus: Corpus, records: int, dimensions: int) -> Measured:
    """Searches in a file of ``records`` records in one namespace, with vectors."""
    path = folder / f"{records}-{dimensions}.db"
    with MemoryEngine.sqlite(path, embedder=random_model(dimensions)) as engine:
        notes = filled(engine, corpus, "user:1", records)
        rare = fastest(lambda: notes.search(RARE, limit=LIMIT, mode="text"))
        common = fastest(lambda: notes.search(corpus.common, limit=LIMIT, mode="text"))
        meaning = fastest(lambda: notes.search(corpus.common, limit=LIMIT, mode="vector"))
        mixed = fastest(lambda: notes.search(corpus.common, limit=LIMIT))
        memory = peak_memory(lambda: notes.search(corpus.common, limit=LIMIT, mode="vector"))
        new = corpus.texts(RUNS, seed=-1)
        dedupe = fastest(lambda: notes.add(next(new), dedupe=True))
    file_size = path.stat().st_size  # once closed, so that the write-ahead log is in the file
    path.unlink()
    return Measured(rare, common, meaning, mixed, dedupe, memory, file_size)


def many_namespaces(folder: Path, corpus: Corpus) -> tuple[float, float]:
    """Searches by words in one namespace of a file of many, as (a rare word, common words)."""
    path = folder / "namespaces.db"
    with MemoryEngine.sqlite(path) as engine:
        notes = filled(engine, corpus, "user:0", PER_NAMESPACE)
        for number in range(1, NAMESPACES):
            fill(engine.collection(f"user:{number}"), corpus.texts(PER_NAMESPACE, seed=number))
        found = (
            fastest(lambda: notes.search(RARE, limit=LIMIT, mode="text")),
            fastest(lambda: notes.search(corpus.common, limit=LIMIT, mode="text")),
        )
    path.unlink()
    return found


def once(action: Callable[[], object]) -> float:
    """How long ``action`` takes to run once, in seconds."""
    started = time.perf_counter()
    action()
    return time.perf_counter() - started


def hidden(folder: Path, corpus: Corpus) -> tuple[float, float, float, float]:
    """In one namespace of ``HIDDEN`` records, every other one expired: ``count()``, the first
    page of ``list()``, then ``purge_expired()`` and ``erase_deleted()``, which run once."""
    path = folder / "hidden.db"
    with MemoryEngine.sqlite(path) as engine:
        notes = engine.collection("user:1")
        for batch in batched(enumerate(corpus.texts(HIDDEN)), BATCH):
            notes.add_many(
                NewMemory(text, expires_at=EXPIRED_AT if number % 2 else None)
                for number, text in batch
            )
        count = fastest(notes.count)
        page = fastest(lambda: notes.list(limit=LIMIT))
        purge = once(engine.purge_expired)
        erase = once(engine.erase_deleted)
    path.unlink()
    return count, page, purge, erase


def awaiting() -> float:
    """How much longer ``count()`` takes awaited than called straight, in seconds per call."""
    with MemoryEngine.in_memory() as engine:
        notes = engine.collection("user:1")
        straight = fastest(lambda: [notes.count() for _ in range(ASYNC_CALLS)])

    async def counts(collection: AsyncCollection) -> None:
        for _ in range(ASYNC_CALLS):
            await collection.count()

    with asyncio.Runner() as runner:
        twin = runner.run(AsyncMemoryEngine.in_memory())
        collection = twin.collection("user:1")
        awaited = fastest(lambda: runner.run(counts(collection)))
        runner.run(twin.close())
    return (awaited - straight) / ASYNC_CALLS


def first_open(folder: Path, corpus: Corpus, version: int) -> float:
    """The first open of a file at schema ``version``, which brings it up to date."""
    path = folder / f"version-{version}.db"
    make_old_file(
        path, version, lambda engine: fill(engine.collection("user:1"), corpus.texts(UPGRADED))
    )
    started = time.perf_counter()
    engine = MemoryEngine.sqlite(path)
    opened = time.perf_counter() - started
    engine.close()
    path.unlink()
    return opened


def duration(seconds: float) -> str:
    if seconds >= 1:
        return f"{seconds:.1f} s"
    milliseconds = seconds * 1000
    return f"{milliseconds:.1f} ms" if milliseconds < 10 else f"{milliseconds:.0f} ms"


def size(count: int) -> str:
    return f"{count / 1e9:.1f} GB" if count >= 1e9 else f"{count / 1e6:,.0f} MB"


def main() -> None:
    print(
        f"pybodn {__version__} on {platform.platform()}: Python {platform.python_version()},"
        f" SQLite {sqlite3.sqlite_version}, NumPy {np.__version__}. Measuring...\n",
        flush=True,
    )
    corpus = Corpus()
    with tempfile.TemporaryDirectory() as name:
        folder = Path(name)
        found = {
            (records, dimensions): one_namespace(folder, corpus, records, dimensions)
            for records in RECORDS
            for dimensions in DIMENSIONS
        }
        rare, common = many_namespaces(folder, corpus)
        count, page, purge, erase = hidden(folder, corpus)
        opened = {
            release: first_open(folder, corpus, version) for release, version in OLDER.items()
        }
    awaited = awaiting()

    chosen = {records: found[records, MAIN_DIMENSIONS] for records in RECORDS}
    rows = [
        ("By words: a word that one record holds", [chosen[n].rare for n in RECORDS]),
        (f"By words: the {COMMON_WORDS} most common words", [chosen[n].common for n in RECORDS]),
        *(
            (
                f"By meaning: vectors of {dimensions:,} numbers",
                [found[n, dimensions].meaning for n in RECORDS],
            )
            for dimensions in DIMENSIONS
        ),
        (
            f"Mixed: {MAIN_DIMENSIONS:,} numbers and the {COMMON_WORDS} most common words",
            [chosen[n].mixed for n in RECORDS],
        ),
    ]
    columns = " | ".join(f"{records:,} records" for records in RECORDS)
    print(f"| Search, best {LIMIT} | {columns} |")
    print(f"|---{'|---' * len(RECORDS)}|")
    for label, times in rows:
        print(f"| {label} | {' | '.join(map(duration, times))} |")
    memory = " and ".join(f"{size(chosen[records].memory)} at {records:,}" for records in RECORDS)
    dedupe = " and ".join(
        f"{duration(chosen[records].dedupe)} at {records:,}" for records in RECORDS
    )
    upgrades = ", ".join(f"from {release} {duration(opened[release])}" for release in OLDER)
    print(
        f"\nMemory of a search by meaning, vectors of {MAIN_DIMENSIONS:,} numbers: {memory}."
        f"\nFile of {RECORDS[-1]:,} records with vectors of {MAIN_DIMENSIONS:,} numbers:"
        f" {size(chosen[RECORDS[-1]].file_size)}."
        f"\nSearch by words in one namespace of a file of {NAMESPACES * PER_NAMESPACE:,} records"
        f" in {NAMESPACES:,} namespaces: {duration(rare)} for a word that one record holds,"
        f" {duration(common)} for the {COMMON_WORDS} most common words."
        f"\nAdd of a new text with dedupe=True, in one namespace: {dedupe}."
        f"\nOne namespace of {HIDDEN:,} records, every other one expired:"
        f" count() {duration(count)}, the first page of list() {duration(page)},"
        f" purge_expired() of {HIDDEN // 2:,} records {duration(purge)},"
        f" then erase_deleted() {duration(erase)}."
        f"\nFirst open of a file of {UPGRADED:,} records from an older pybodn: {upgrades}."
        f"\nAn awaited call takes {awaited * 1e6:.0f} µs longer than the same call straight."
    )


if __name__ == "__main__":
    main()
