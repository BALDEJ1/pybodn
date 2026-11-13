import math
import threading
from collections.abc import Callable
from enum import IntEnum, StrEnum
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType

import pytest

from pybodn import _validation as validate
from pybodn._models import NewMemory
from pybodn.errors import ValidationError

MAX_INTEGER = 2**63 - 1
MIN_INTEGER = -(2**63)
LONE_SURROGATE = "\ud800"
WAIT_SECONDS = 10  # a check that never ends fails after this instead of hanging
MAX_TIMEOUT = 2_147_483  # seconds: SQLite counts the wait in milliseconds, in a 32-bit integer


class Color(StrEnum):
    RED = "red"


class Huge(IntEnum):
    PAST_64_BITS = 2**63


class Unencodable(str):
    """Text that fails the test if anything encodes it."""

    def encode(self, encoding: str = "utf-8", errors: str = "strict") -> bytes:
        raise AssertionError("the text was encoded")


def circular() -> dict[str, object]:
    value: dict[str, object] = {}
    value["self"] = value
    return value


def nested(levels: int) -> dict[str, object]:
    """A dict holding lists, ``levels`` containers deep in total."""
    value: object = "leaf"
    for _ in range(levels - 1):
        value = [value]
    return {"deep": value}


def error_within_time_limit(check: Callable[[], object]) -> Exception:
    """Run ``check`` in a thread, so a check that never ends fails the test; return its error."""
    errors: list[Exception] = []

    def run() -> None:
        try:
            check()
        except Exception as error:
            errors.append(error)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout=WAIT_SECONDS)
    assert not worker.is_alive(), f"the check did not finish in {WAIT_SECONDS} seconds"
    assert len(errors) == 1, "the check raised nothing"
    return errors[0]


# text: namespace, content, id ---------------------------------------------------------------


@pytest.mark.parametrize(
    "value", ["a", "Prefers Python", "юнікод і емодзі 🙂", "nul\x00inside", "x" * 100_000]
)
def test_text_accepts_non_empty_strings(value: str) -> None:
    assert validate.text("content", value) == value


@pytest.mark.parametrize("value", [None, 1, b"bytes", ["a"], "", f"bad {LONE_SURROGATE}"])
def test_text_rejects_anything_else_and_names_the_field(value: object) -> None:
    with pytest.raises(ValidationError, match="content"):
        validate.text("content", value)


def test_text_returns_a_plain_str_for_a_str_subclass() -> None:
    # Act
    result = validate.text("key", Color.RED)

    # Assert
    assert type(result) is str
    assert result == "red"


# key and source ------------------------------------------------------------------------------


@pytest.mark.parametrize("value", [None, "language"])
def test_key_accepts_none_or_non_empty_string(value: str | None) -> None:
    assert validate.key(value) == value


@pytest.mark.parametrize("value", ["", 1, LONE_SURROGATE])
def test_key_rejects_anything_else(value: object) -> None:
    with pytest.raises(ValidationError, match="key"):
        validate.key(value)


@pytest.mark.parametrize("value", [None, "", "chat"])
def test_source_accepts_none_or_any_string(value: str | None) -> None:
    assert validate.source(value) == value


@pytest.mark.parametrize("value", [1, LONE_SURROGATE])
def test_source_rejects_anything_else(value: object) -> None:
    with pytest.raises(ValidationError, match="source"):
        validate.source(value)


# importance ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"), [(0, 0.0), (1, 1.0), (0.5, 0.5), (Fraction(1, 4), 0.25)]
)
def test_importance_returns_a_float(value: object, expected: float) -> None:
    # Act
    result = validate.importance(value)

    # Assert
    assert type(result) is float
    assert result == expected


def test_importance_turns_negative_zero_into_zero() -> None:
    # SQLite reads -0.0 back as 0.0, so both storages must return 0.0.
    assert math.copysign(1.0, validate.importance(-0.0)) == 1.0


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(True, id="true"),
        pytest.param(False, id="false"),
        pytest.param(math.nan, id="nan"),
        pytest.param(math.inf, id="inf"),
        pytest.param(-0.1, id="below-zero"),
        pytest.param(1.1, id="above-one"),
        pytest.param(10**400, id="int-too-large-for-float"),
        pytest.param("0.5", id="string"),
        pytest.param(None, id="none"),
    ],
)
def test_importance_rejects_anything_else(value: object) -> None:
    with pytest.raises(ValidationError, match="importance"):
        validate.importance(value)


# metadata ------------------------------------------------------------------------------------


def test_metadata_returns_an_equal_dict() -> None:
    # Arrange
    original = {"text": "юнікод 🙂\x00", "number": 1, "float": 2.5, "flag": True, "none": None}

    # Act
    result = validate.metadata(MappingProxyType(original))

    # Assert
    assert type(result) is dict
    assert result == original


def test_metadata_is_a_deep_copy() -> None:
    # Arrange
    inner = [1, 2]
    original = {"nested": {"list": inner}}

    # Act
    result = validate.metadata(original)
    inner.append(3)

    # Assert
    assert result == {"nested": {"list": [1, 2]}}


def test_metadata_keeps_the_key_order() -> None:
    assert list(validate.metadata({"b": 1, "a": 2})) == ["b", "a"]


def test_metadata_allows_100_levels_of_nesting() -> None:
    assert validate.metadata(nested(100)) == nested(100)


def test_metadata_allows_values_up_to_the_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    monkeypatch.setattr(validate, "_MAX_VALUES", 5)
    value = {"list": [1, 2, 3]}  # the dict, the list and three numbers

    # Act / Assert
    assert validate.metadata(value) == value


def test_metadata_rejects_one_value_over_the_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    monkeypatch.setattr(validate, "_MAX_VALUES", 5)

    # Act / Assert
    with pytest.raises(ValidationError, match="metadata must not hold more than 5 values"):
        validate.metadata({"list": [1, 2, 3, 4]})


def test_metadata_that_reuses_one_list_over_and_over_is_rejected() -> None:
    # Arrange: small in memory, but over a billion values once written out in full
    value: list[object] = []
    for _ in range(30):
        value = [value, value]

    # Act
    error = error_within_time_limit(lambda: validate.metadata({"value": value}))

    # Assert
    assert isinstance(error, ValidationError)
    assert "1,000,000 values" in str(error)


@pytest.mark.parametrize(
    ("text", "size"),
    [
        pytest.param("abcdefgh", 10, id="ascii"),
        pytest.param("їїїї", 10, id="two-byte-letters"),
        pytest.param("🙂🙂", 10, id="four-byte-emoji"),
    ],
)
def test_metadata_allows_text_up_to_the_limit_in_utf8_bytes(
    monkeypatch: pytest.MonkeyPatch, text: str, size: int
) -> None:
    # Arrange: the key "ab" takes two bytes, the text the rest
    monkeypatch.setattr(validate, "_MAX_TEXT_BYTES", size)

    # Act / Assert
    assert validate.metadata({"ab": text}) == {"ab": text}


@pytest.mark.parametrize(
    "value",
    [
        pytest.param({"ab": "abcdefghi"}, id="value-one-byte-over"),
        pytest.param({"ab": "їїїїa"}, id="letters-counted-in-bytes"),
        pytest.param({"abcdefghijk": 1}, id="long-key"),
        pytest.param({"abcdef": 1, "ghijk": 2}, id="keys-added-up"),
        pytest.param({"a": ["abcde", "abcde"]}, id="texts-added-up"),
    ],
)
def test_metadata_rejects_text_over_the_limit(
    monkeypatch: pytest.MonkeyPatch, value: dict[str, object]
) -> None:
    # Arrange
    monkeypatch.setattr(validate, "_MAX_TEXT_BYTES", 10)

    # Act / Assert
    with pytest.raises(ValidationError, match="metadata must not hold more than 10 bytes of text"):
        validate.metadata(value)


def test_metadata_rejects_text_with_more_characters_than_the_limit_without_encoding_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange: every character takes at least a byte, so the size is known without encoding
    monkeypatch.setattr(validate, "_MAX_TEXT_BYTES", 10)

    # Act / Assert
    with pytest.raises(ValidationError, match="10 bytes of text"):
        validate.metadata({"a": Unencodable("abcdefghijk")})


LONG_TEXT = "x" * 1_000_000


@pytest.mark.parametrize(
    "value",
    [
        pytest.param({"list": [LONG_TEXT] * 1_000}, id="as-a-value"),
        pytest.param({"list": [{LONG_TEXT: 1}] * 1_000}, id="as-a-key"),
    ],
)
def test_metadata_that_reuses_one_long_text_over_and_over_is_rejected(
    value: dict[str, object],
) -> None:
    # Arrange: one megabyte in memory, a gigabyte once written out in full

    # Act
    error = error_within_time_limit(lambda: validate.metadata(value))

    # Assert
    assert isinstance(error, ValidationError)
    assert "16,777,216 bytes of text" in str(error)


@pytest.mark.parametrize("number", [MIN_INTEGER, MAX_INTEGER])
def test_metadata_allows_integers_that_fit_in_64_bits(number: int) -> None:
    assert validate.metadata({"list": [number]}) == {"list": [number]}


@pytest.mark.parametrize(
    "number",
    [
        pytest.param(MIN_INTEGER - 1, id="one-below"),
        pytest.param(MAX_INTEGER + 1, id="one-above"),
        pytest.param(10**4000, id="thousands-of-digits"),
        pytest.param(Huge.PAST_64_BITS, id="int-subclass"),
    ],
)
def test_metadata_rejects_integers_that_do_not_fit_in_64_bits(number: int) -> None:
    with pytest.raises(
        ValidationError, match=f"metadata integers must be from {MIN_INTEGER} to {MAX_INTEGER}"
    ):
        validate.metadata({"list": [number]})


def test_metadata_that_reuses_one_huge_integer_over_and_over_is_rejected() -> None:
    # Arrange: a few kilobytes in memory, gigabytes once written out in full, and
    # 1,000,000 values with the dict and the list, so the value limit lets it through
    value = {"list": [10**4000] * 999_998}

    # Act
    error = error_within_time_limit(lambda: validate.metadata(value))

    # Assert
    assert isinstance(error, ValidationError)
    assert "metadata integers" in str(error)


def test_metadata_turns_str_subclasses_into_plain_strs() -> None:
    # Act
    result = validate.metadata({"color": Color.RED})

    # Assert
    assert result == {"color": "red"}
    assert type(result["color"]) is str


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(["not", "a", "mapping"], id="list"),
        pytest.param(None, id="none"),
        pytest.param({1: "value"}, id="number-key"),
        pytest.param({"nested": {2: "value"}}, id="nested-number-key"),
        pytest.param({"1": "first", 1: "second"}, id="keys-that-collide-as-text"),
        pytest.param({"value": math.nan}, id="nan"),
        pytest.param({"value": math.inf}, id="inf"),
        pytest.param({"value": {1, 2}}, id="set"),
        pytest.param({"value": (1, 2)}, id="tuple"),
        pytest.param({"value": 10**5000}, id="int-too-large-for-text"),
        pytest.param({"value": LONE_SURROGATE}, id="lone-surrogate-value"),
        pytest.param({LONE_SURROGATE: "value"}, id="lone-surrogate-key"),
        pytest.param(circular(), id="circular"),
        pytest.param(nested(101), id="101-levels"),
    ],
)
def test_metadata_rejects_anything_but_json_with_string_keys(value: object) -> None:
    with pytest.raises(ValidationError, match="metadata"):
        validate.metadata(value)


# tags ----------------------------------------------------------------------------------------


def test_tags_become_a_tuple_without_duplicates_in_first_seen_order() -> None:
    assert validate.tags(["b", "a", "b"]) == ("b", "a")


def test_tags_accept_any_iterable_of_strings() -> None:
    assert validate.tags(iter(("a", Color.RED))) == ("a", "red")


def test_tags_are_plain_strs() -> None:
    assert [type(tag) for tag in validate.tags([Color.RED])] == [str]


@pytest.mark.parametrize("value", ["pref", None, 5, [""], [1], [LONE_SURROGATE]])
def test_tags_reject_anything_else(value: object) -> None:
    with pytest.raises(ValidationError, match="tag"):
        validate.tags(value)


# limit and offset ----------------------------------------------------------------------------


@pytest.mark.parametrize("value", [None, 1, MAX_INTEGER])
def test_limit_accepts_none_or_an_integer_from_one(value: int | None) -> None:
    assert validate.limit(value) == value


@pytest.mark.parametrize("value", [0, -1, MAX_INTEGER + 1, True, 1.0, "1"])
def test_limit_rejects_anything_else(value: object) -> None:
    with pytest.raises(ValidationError, match="limit"):
        validate.limit(value)


@pytest.mark.parametrize("value", [0, MAX_INTEGER])
def test_offset_accepts_an_integer_from_zero(value: int) -> None:
    assert validate.offset(value) == value


@pytest.mark.parametrize("value", [-1, MAX_INTEGER + 1, None, False, 0.0])
def test_offset_rejects_anything_else(value: object) -> None:
    with pytest.raises(ValidationError, match="offset"):
        validate.offset(value)


# expected_version ----------------------------------------------------------------------------


@pytest.mark.parametrize("value", [None, 1, MAX_INTEGER])
def test_expected_version_accepts_none_or_an_integer_from_one(value: int | None) -> None:
    assert validate.expected_version(value) == value


@pytest.mark.parametrize("value", [0, -1, MAX_INTEGER + 1, True, 1.0, "1"])
def test_expected_version_rejects_anything_else(value: object) -> None:
    with pytest.raises(ValidationError, match="expected_version"):
        validate.expected_version(value)


# memories ------------------------------------------------------------------------------------


def test_memories_become_a_list_from_any_iterable() -> None:
    batch = (NewMemory(text) for text in ("a", "b"))
    assert validate.memories(batch) == [NewMemory("a"), NewMemory("b")]


@pytest.mark.parametrize("value", [None, 5, "text", [None], [{"content": "text"}]])
def test_memories_reject_anything_but_new_memories(value: object) -> None:
    with pytest.raises(ValidationError, match="memories"):
        validate.memories(value)


# path ----------------------------------------------------------------------------------------


class BytesPath:
    """A path object that gives bytes."""

    def __fspath__(self) -> bytes:
        return b"memory.db"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param("memory.db", "memory.db", id="str"),
        pytest.param(
            Path("folder", "пам'ять 🙂.db"), str(Path("folder", "пам'ять 🙂.db")), id="path"
        ),
        pytest.param(":memory:", ":memory:", id="in-memory"),
        pytest.param(Color.RED, "red", id="str-subclass"),
    ],
)
def test_path_accepts_a_string_or_a_path_object(value: object, expected: str) -> None:
    # Act
    result = validate.path(value)

    # Assert
    assert result == expected
    assert type(result) is str


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("", id="empty"),  # SQLite would make a temporary database, lost on close
        pytest.param("memory\x00.db", id="nul"),
        pytest.param("\ud800.db", id="lone-surrogate"),
        pytest.param(b"memory.db", id="bytes"),
        pytest.param(BytesPath(), id="path-object-giving-bytes"),
        pytest.param(None, id="none"),
        pytest.param(1, id="number"),
    ],
)
def test_path_rejects_anything_else(value: object) -> None:
    with pytest.raises(ValidationError, match="path"):
        validate.path(value)


# timeout -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, 0.0),
        (-0.0, 0.0),
        (0.5, 0.5),
        (5, 5.0),
        (Fraction(1, 4), 0.25),
        (MAX_TIMEOUT, MAX_TIMEOUT),
    ],
)
def test_timeout_accepts_seconds_and_returns_a_float(value: object, expected: float) -> None:
    # Act
    result = validate.timeout(value)

    # Assert
    assert result == expected
    assert type(result) is float


@pytest.mark.parametrize(
    "value",
    [-0.001, -1, MAX_TIMEOUT + 0.001, math.nan, math.inf, True, "5", None, 10**400],
)
def test_timeout_rejects_anything_else(value: object) -> None:
    with pytest.raises(ValidationError, match="timeout"):
        validate.timeout(value)
