import tomllib
from pathlib import Path

import pybodn

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def test_version_matches_pyproject() -> None:
    # Arrange
    expected = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]["version"]

    # Act
    actual = pybodn.__version__

    # Assert
    assert actual == expected
