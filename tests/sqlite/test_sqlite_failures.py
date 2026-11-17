"""A write that fails at any file action leaves the file as its last commit before that left it.

Each write runs on a copy of one file again and again, its first file action failing, then its
second, and so on through its last. Every action is reached this way, new ones too, without a
list of them to keep up to date.
"""

import shutil
import sqlite3
from collections.abc import Callable
from contextlib import closing
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest

from pybodn import Collection, Embedder, MemoryEngine, NewMemory
from pybodn._storage.sqlite import SQLiteStore
from pybodn.errors import StorageError
from tests.fake_model import FakeModel

pytest.importorskip("numpy")  # the writes give records vectors

NOW = datetime(2026, 11, 17, 12, 0, tzinfo=UTC)
PAST = NOW - timedelta(days=1)
MODEL = "fake-1"
BATCH = 2  # records per batch of embed_missing and purge_expired, so that three take two


type File = list[str]  # the file's whole content, as SQL that would make it again


def file_of(path: Path) -> File:
    with closing(sqlite3.connect(path)) as connection:
        return [*connection.iterdump()]


class Actions:
    """Stands in for the store's connection and numbers its file actions, every call of
    ``execute`` or ``executemany``, from 1.

    Action ``fail_at`` fails, as it would on a full disk, and every other passes. Without it,
    nothing fails, and ``after_commit`` keeps the file as each COMMIT left it, by its number.
    """

    def __init__(self, connection: sqlite3.Connection, path: Path, fail_at: int | None) -> None:
        self._connection = connection
        self._path = path
        self._fail_at = fail_at
        self.count = 0
        self.after_commit: dict[int, File] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._connection, name)

    def execute(self, sql: str, *args: Any) -> sqlite3.Cursor:
        self._next()
        cursor = self._connection.execute(sql, *args)
        if sql == "COMMIT" and self._fail_at is None:
            self.after_commit[self.count] = file_of(self._path)
        return cursor

    def executemany(self, sql: str, *args: Any) -> sqlite3.Cursor:
        self._next()
        return self._connection.executemany(sql, *args)

    def _next(self) -> None:
        self.count += 1
        if self.count == self._fail_at:
            raise sqlite3.OperationalError("database or disk is full")


def engine_on(store: SQLiteStore, *, with_model: bool = True) -> MemoryEngine:
    embedder = Embedder(MODEL, FakeModel(), batch_size=BATCH) if with_model else None
    return MemoryEngine(store, clock=lambda: NOW, embedder=embedder)


def notes_of(engine: MemoryEngine) -> Collection:
    return engine.collection("user:123")


@pytest.fixture
def base(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A closed file that holds every kind of record a write may reach."""
    monkeypatch.setattr("pybodn._engine._PURGE_BATCH", BATCH)
    path = tmp_path / "base.db"
    store = SQLiteStore(str(path), timeout=5.0)
    with engine_on(store, with_model=False) as engine:
        for text in ["чай", "молоко", "арахіс"]:  # no vectors yet
            notes_of(engine).add(text)
    store = SQLiteStore(str(path), timeout=5.0)
    with engine_on(store) as engine:
        notes = notes_of(engine)
        notes.put("drink", "кава з молоком", tags=["drink"], metadata={"sugar": False})
        notes.add("python і кава")
        notes.archive(notes.add("стара кава", key="old").id)
        for text in ["кава вчора", "чай вчора", "молоко вчора"]:
            notes.add(text, expires_at=PAST)
    return path


def id_of(notes: Collection, key: str) -> str:
    record = notes.get_by_key(key, include_archived=True)
    assert record is not None
    return record.id


# Each write, and how many times it changes the file: a batch of a few records at a time
# commits each batch on its own.
WRITES: dict[str, tuple[Callable[[MemoryEngine], object], int]] = {
    "add": (
        lambda engine: notes_of(engine).add(
            "арахіс і алергія", key="allergy", tags=["health"], metadata={"level": 2}
        ),
        1,
    ),
    "add-many": (
        lambda engine: notes_of(engine).add_many([NewMemory("чай"), NewMemory("кава", key="k")]),
        1,
    ),
    "add-many-with-dedupe": (
        lambda engine: notes_of(engine).add_many(
            [NewMemory("python і кава"), NewMemory("кава")], dedupe=True
        ),
        1,
    ),
    "put-new": (lambda engine: notes_of(engine).put("food", "арахіс"), 1),
    "put-over": (
        lambda engine: notes_of(engine).put("drink", "чай з молоком", tags=["tea"]),
        1,
    ),
    "update-text": (
        lambda engine: notes_of(engine).update(
            id_of(notes_of(engine), "drink"), content="кава без молока"
        ),
        1,
    ),
    "update-fields": (
        lambda engine: notes_of(engine).update(
            id_of(notes_of(engine), "drink"), metadata={"sugar": True}, importance=0.9
        ),
        1,
    ),
    "delete": (lambda engine: notes_of(engine).delete(id_of(notes_of(engine), "drink")), 1),
    "archive": (lambda engine: notes_of(engine).archive(id_of(notes_of(engine), "drink")), 1),
    "restore": (lambda engine: notes_of(engine).restore(id_of(notes_of(engine), "old")), 1),
    "purge-expired": (lambda engine: engine.purge_expired(), 2),
    "embed-missing": (lambda engine: engine.embed_missing(), 2),
}


def run(
    base: Path, path: Path, write: Callable[[MemoryEngine], object], fail_at: int | None
) -> Actions:
    """Run ``write`` on a copy of ``base`` at ``path``, its action ``fail_at`` failing."""
    shutil.copyfile(base, path)
    store = SQLiteStore(str(path), timeout=5.0)
    actions = Actions(store._connection, path, fail_at)
    store._connection = actions  # type: ignore[assignment]
    with engine_on(store) as engine:
        if fail_at is None:
            write(engine)
        else:
            with pytest.raises(StorageError) as caught:
                write(engine)
            assert isinstance(caught.value.__cause__, sqlite3.OperationalError)
    return actions


def changes(before: File, files: list[File]) -> int:
    """How many times the file changed, going from ``before`` through ``files``."""
    return sum(1 for older, newer in pairwise([before, *files]) if older != newer)


@pytest.mark.parametrize(("write", "times"), WRITES.values(), ids=WRITES.keys())
def test_a_write_failing_at_any_action_leaves_the_file_as_its_last_commit_before(
    base: Path, tmp_path: Path, write: Callable[[MemoryEngine], object], times: int
) -> None:
    # Arrange: a run without failures numbers the actions and keeps the file after each commit.
    # Every change it makes comes with a commit, so the file ends as the last commit left it.
    before = file_of(base)
    clean = run(base, tmp_path / "clean.db", write, fail_at=None)
    assert changes(before, [*clean.after_commit.values(), file_of(tmp_path / "clean.db")]) == times

    for fail_at in range(1, clean.count + 1):
        # Act
        run(base, tmp_path / f"{fail_at}.db", write, fail_at=fail_at)

        # Assert
        committed = [number for number in clean.after_commit if number < fail_at]
        expected = clean.after_commit[committed[-1]] if committed else before
        assert file_of(tmp_path / f"{fail_at}.db") == expected, f"action {fail_at}"
