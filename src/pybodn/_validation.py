"""Every input check in one place, so all storages receive the same clean values.

Each function takes any value and returns it in its stored form, or raises
``ValidationError`` with a message that names the field.
"""

import json
import numbers
import os
from collections.abc import Iterable, Mapping
from typing import NamedTuple

from pybodn._models import JSONValue, NewMemory
from pybodn.errors import ValidationError

_MAX_INTEGER = 2**63 - 1  # the largest integer SQLite can store
_MIN_INTEGER = -(2**63)  # the smallest
_MAX_TIMEOUT = 2_147_483  # seconds: SQLite counts the wait in milliseconds, in a 32-bit integer

# How deep Python itself can nest depends on the platform's stack, so data written on
# one machine could fail to load on another. A fixed limit behaves the same everywhere.
_MAX_DEPTH = 100

# Metadata is measured as it would be written out. The same list placed inside itself
# twice, again and again, looks tiny but writes out as billions of values; one long text
# placed many times writes out in full each time. 16 MB is what MongoDB allows a document.
# Integers are held to what SQLite can store, so none of them writes out longer than 20
# characters.
_MAX_VALUES = 1_000_000
_MAX_TEXT_BYTES = 16 * 1024 * 1024


class _Size(NamedTuple):
    values: int
    text_bytes: int


def text(name: str, value: object) -> str:
    """A non-empty string: ``namespace``, ``content``, a key, a tag or an id."""
    result = _string(name, value)
    if not result:
        raise ValidationError(f"{name} must not be empty")
    return result


def key(value: object) -> str | None:
    """No key, or a non-empty string."""
    return None if value is None else text("key", value)


def source(value: object) -> str | None:
    """No source, or any string."""
    return None if value is None else _string("source", value)


def importance(value: object) -> float:
    """A real number from 0 to 1, as ``float``. ``bool`` and NaN are rejected."""
    number = _as_float(value)
    if number is None or not 0.0 <= number <= 1.0:  # NaN fails the range check too
        raise ValidationError("importance must be a number from 0 to 1")
    return abs(number)  # SQLite reads -0.0 back as 0.0


def metadata(value: object) -> dict[str, JSONValue]:
    """A deep copy: JSON values under string keys, at most 100 levels deep.

    At most 1,000,000 values and 16 MB of text in all, counted as written out: the
    mapping and every list and dict are one value each, keys count as text in UTF-8,
    and anything placed twice counts twice. Integers must fit in 64 bits, as SQLite
    stores them.

    Non-string keys and tuples are rejected rather than converted, because JSON
    would silently turn ``{1: ...}`` into ``{"1": ...}`` and a tuple into a list.
    """
    if not isinstance(value, Mapping):
        raise ValidationError(f"metadata must be a mapping, not {type(value).__name__}")
    original = dict(value)
    _check_nesting(original, levels_left=_MAX_DEPTH, room=_Size(_MAX_VALUES, _MAX_TEXT_BYTES))
    try:
        encoded = json.dumps(original, ensure_ascii=False, allow_nan=False)
        encoded.encode("utf-8")  # lone surrogates cannot be stored
    except (TypeError, ValueError) as error:
        raise ValidationError(f"metadata must contain only JSON values: {error}") from error
    copy: dict[str, JSONValue] = json.loads(encoded)
    return copy


def tags(value: object) -> tuple[str, ...]:
    """Non-empty strings, without duplicates, in first-seen order."""
    if isinstance(value, str) or not isinstance(value, Iterable):
        raise ValidationError("tags must be a list of strings")
    return tuple(dict.fromkeys(text("tag", tag) for tag in value))


def limit(value: object) -> int | None:
    """No limit, or an integer from 1."""
    return None if value is None else _integer("limit", value, minimum=1)


def offset(value: object) -> int:
    """An integer from 0."""
    return _integer("offset", value, minimum=0)


def expected_version(value: object) -> int | None:
    """No expected version, or an integer from 1. ``True`` is not taken for 1."""
    return None if value is None else _integer("expected_version", value, minimum=1)


def memories(value: object) -> list[NewMemory]:
    """A batch of ``NewMemory`` objects from any iterable, as a list."""
    if not isinstance(value, Iterable):
        raise ValidationError(f"memories must be an iterable, not {type(value).__name__}")
    batch = list(value)
    for memory in batch:
        if not isinstance(memory, NewMemory):
            raise ValidationError(
                f"memories must contain NewMemory objects, not {type(memory).__name__}"
            )
    return batch


def path(value: object) -> str:
    """A database path: a non-empty string or a path object, as ``str``."""
    result = text("path", os.fspath(value) if isinstance(value, os.PathLike) else value)
    if "\x00" in result:
        raise ValidationError("path must not contain a NUL character")
    return result


def timeout(value: object) -> float:
    """Seconds to wait, as ``float``: a real number from 0. ``bool`` and NaN are rejected."""
    number = _as_float(value)
    if number is None or not 0.0 <= number <= _MAX_TIMEOUT:
        raise ValidationError(f"timeout must be a number of seconds from 0 to {_MAX_TIMEOUT}")
    return number


def _string(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"{name} must be a string, not {type(value).__name__}")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise ValidationError(f"{name} must not contain lone surrogates") from None
    # A plain str with the same characters: a str-based enum would otherwise come
    # back as itself from memory but as plain text from SQLite.
    return str.__str__(value)


def _as_float(value: object) -> float | None:
    """``value`` as a float if it is a real number other than ``bool`` that fits one."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    try:
        return float(value)
    except OverflowError:  # an integer too large for a float
        return None


def _check_nesting(value: object, *, levels_left: int, room: _Size) -> _Size:
    """Allow dicts with string keys and lists, at most ``levels_left`` levels deep and no
    larger than ``room``. Return the size of ``value``, itself included.

    A structure that contains itself is infinitely deep, so it is rejected too.
    """
    if room.values == 0:
        raise ValidationError(f"metadata must not hold more than {_MAX_VALUES:,} values")
    if isinstance(value, tuple):
        raise ValidationError("metadata must use lists, not tuples")
    if isinstance(value, str):
        return _Size(1, _text_bytes(value, room.text_bytes))
    if isinstance(value, int) and not _MIN_INTEGER <= value <= _MAX_INTEGER:
        raise ValidationError(f"metadata integers must be from {_MIN_INTEGER} to {_MAX_INTEGER}")
    if not isinstance(value, dict | list):
        return _Size(1, 0)  # json.dumps rejects anything that is not a JSON value
    if levels_left == 0:
        raise ValidationError(f"metadata must not be nested more than {_MAX_DEPTH} levels deep")
    values, text_bytes = 1, 0
    children: Iterable[object]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValidationError("metadata keys must be strings")
        for key in value:
            text_bytes += _text_bytes(key, room.text_bytes - text_bytes)
        children = value.values()
    else:
        children = value
    for child in children:
        left = _Size(room.values - values, room.text_bytes - text_bytes)
        size = _check_nesting(child, levels_left=levels_left - 1, room=left)
        values += size.values
        text_bytes += size.text_bytes
    return _Size(values, text_bytes)


def _text_bytes(text: str, bytes_left: int) -> int:
    """The size of ``text`` in UTF-8, if it fits in ``bytes_left``."""
    # Every character takes at least a byte, so a text with more characters cannot fit:
    # no need to encode it. Lone surrogates are rejected later, when the whole is encoded.
    too_long = len(text) > bytes_left
    size = len(text) if too_long else len(text.encode("utf-8", "surrogatepass"))
    if size > bytes_left:
        raise ValidationError(f"metadata must not hold more than {_MAX_TEXT_BYTES:,} bytes of text")
    return size


def _integer(name: str, value: object, *, minimum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= _MAX_INTEGER
    ):
        raise ValidationError(f"{name} must be an integer from {minimum} to {_MAX_INTEGER}")
    return int(value)
