"""Every input check in one place, so all storages receive the same clean values.

Each function takes any value and returns it in its stored form, or raises
``ValidationError`` with a message that names the field.
"""

import json
import numbers
from collections.abc import Iterable, Mapping

from pybodn._models import JSONValue
from pybodn.errors import ValidationError

_MAX_INTEGER = 2**63 - 1  # the largest integer SQLite can store


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
    """A deep copy that survives a JSON round trip unchanged.

    Unchanged means string keys only and lists rather than tuples: JSON would
    silently turn ``{1: ...}`` into ``{"1": ...}`` and a tuple into a list.
    """
    if not isinstance(value, Mapping):
        raise ValidationError(f"metadata must be a mapping, not {type(value).__name__}")
    original = dict(value)
    try:
        encoded = json.dumps(original, ensure_ascii=False, allow_nan=False)
        encoded.encode("utf-8")  # lone surrogates cannot be stored
        copy: dict[str, JSONValue] = json.loads(encoded)
        is_unchanged = copy == original
    except (TypeError, ValueError, RecursionError) as error:
        raise ValidationError(f"metadata must contain only JSON values: {error}") from error
    if not is_unchanged:
        raise ValidationError("metadata must have only string keys and no tuples")
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


def _integer(name: str, value: object, *, minimum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= _MAX_INTEGER
    ):
        raise ValidationError(f"{name} must be an integer from {minimum} to {_MAX_INTEGER}")
    return int(value)
