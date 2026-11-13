"""Every Python example in the README runs as written, each on its own in an empty folder."""

import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parents[1] / "README.md"
EXAMPLE = re.compile(r"^```python\n(.*?)^```$", re.MULTILINE | re.DOTALL)


def examples() -> list[tuple[int, str]]:
    """Each example with the README line it starts on, so a failure points there."""
    text = README.read_text(encoding="utf-8")
    return [(text.count("\n", 0, found.start(1)) + 1, found[1]) for found in EXAMPLE.finditer(text)]


def test_the_readme_has_examples() -> None:
    assert examples()


@pytest.mark.parametrize(
    ("line", "code"), [pytest.param(*example, id=f"line-{example[0]}") for example in examples()]
)
def test_an_example_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, line: int, code: str
) -> None:
    # Arrange
    monkeypatch.chdir(tmp_path)
    program = compile("\n" * (line - 1) + code, str(README), "exec")

    # Act / Assert
    exec(program, {"__name__": "__main__"})  # noqa: S102
