"""An engine's embedding model: checked before a file is touched; NumPy needed only for it."""

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import pybodn
from pybodn import Embedder, MemoryEngine
from pybodn.errors import ConfigurationError, ValidationError


def model(texts: list[str]) -> list[list[float]]:
    return [[1.0] for _ in texts]


BAD_EMBEDDERS = [
    pytest.param("text-embedding-3-small", "embedder must be an Embedder, not str", id="name"),
    pytest.param(Embedder("", model), "embedder name must not be empty", id="empty-name"),
    pytest.param(
        Embedder(42, model),  # type: ignore[arg-type]
        "embedder name must be a string, not int",
        id="name-not-text",
    ),
    pytest.param(
        Embedder("m", "model"),  # type: ignore[arg-type]
        "embedder embed must be callable",
        id="embed-not-callable",
    ),
    pytest.param(Embedder("m", model, batch_size=0), "batch_size must be an integer", id="0"),
    pytest.param(Embedder("m", model, batch_size=True), "batch_size must be an integer", id="true"),
    pytest.param(
        Embedder("m", model, batch_size=2.0),  # type: ignore[arg-type]
        "batch_size must be an integer",
        id="float",
    ),
]


@pytest.mark.parametrize(("embedder", "message"), BAD_EMBEDDERS)
def test_a_bad_embedder_is_rejected_before_the_file_is_made(
    tmp_path: Path, embedder: Any, message: str
) -> None:
    # Arrange
    path = tmp_path / "memory.db"

    # Act / Assert
    with pytest.raises(ValidationError, match=message):
        MemoryEngine.sqlite(path, embedder=embedder)
    with pytest.raises(ValidationError, match=message):
        MemoryEngine.in_memory(embedder=embedder)
    assert not path.exists()


@pytest.fixture
def without_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    """As if NumPy were not installed."""
    monkeypatch.setitem(sys.modules, "numpy", None)
    monkeypatch.delitem(sys.modules, "pybodn._vectors", raising=False)
    monkeypatch.delattr(pybodn, "_vectors", raising=False)


@pytest.mark.usefixtures("without_numpy")
def test_an_embedder_without_numpy_says_how_to_install_it_before_the_file_is_made(
    tmp_path: Path,
) -> None:
    # Arrange
    path = tmp_path / "memory.db"
    embedder = Embedder("m", model)

    # Act / Assert
    with pytest.raises(ConfigurationError, match=r'pip install "pybodn\[vector\]"') as caught:
        MemoryEngine.sqlite(path, embedder=embedder)
    assert isinstance(caught.value.__cause__, ImportError)
    with pytest.raises(ConfigurationError, match=r'pip install "pybodn\[vector\]"'):
        MemoryEngine.in_memory(embedder=embedder)
    assert not path.exists()


def test_a_numpy_that_is_there_but_fails_to_import_says_why(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: a NumPy that is installed but broken, as one built for another Python
    broken = tmp_path / "site" / "numpy"
    broken.mkdir(parents=True)
    (broken / "__init__.py").write_text('raise ImportError("built for another Python")\n')
    monkeypatch.syspath_prepend(str(tmp_path / "site"))
    monkeypatch.delitem(sys.modules, "numpy", raising=False)
    monkeypatch.delitem(sys.modules, "pybodn._vectors", raising=False)
    monkeypatch.delattr(pybodn, "_vectors", raising=False)

    # Act / Assert: its own error, not advice to install it
    with pytest.raises(ImportError, match="built for another Python"):
        MemoryEngine.in_memory(embedder=Embedder("m", model))


@pytest.mark.usefixtures("without_numpy")
def test_without_numpy_everything_but_search_by_meaning_works(tmp_path: Path) -> None:
    with MemoryEngine.sqlite(tmp_path / "memory.db") as engine:
        # Arrange
        notes = engine.collection("user:123")

        # Act
        record = notes.add("Кава з молоком")

        # Assert
        assert [hit.record for hit in notes.search("кава")] == [record]
        with pytest.raises(ConfigurationError, match="needs an embedding model"):
            notes.search("кава", mode="vector")


def test_numpy_is_never_imported_without_a_model(tmp_path: Path) -> None:
    # Arrange: a fresh interpreter, which has imported nothing yet
    script = (
        "import sys\n"
        "from pybodn import MemoryEngine\n"
        "with MemoryEngine.sqlite(sys.argv[1]) as engine:\n"
        "    notes = engine.collection('user:123')\n"
        "    record = notes.add('Кава')\n"
        "    notes.put('drink', 'Чай')\n"
        "    notes.update(record.id, content='Молоко')\n"
        "    notes.search('кава')\n"
        "    notes.delete(record.id)\n"
        "assert 'numpy' not in sys.modules, 'NumPy was imported'\n"
    )

    # Act / Assert
    subprocess.run(  # noqa: S603 - this interpreter, running the script above
        [sys.executable, "-c", script, str(tmp_path / "memory.db")], check=True
    )
