"""Every input check in one place, so all storages receive the same clean values.

Each function takes any value and returns it in its stored form, or raises
``ValidationError`` with a message that names the field.
"""

import json
import math
import numbers
import os
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from typing import NamedTuple, get_args

from pybodn._models import Embedder, JSONScalar, JSONValue, NewMemory, SearchMode
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

# Each tag and each metadata field of a filter is a condition of its own in one SQL
# expression, and SQLite refuses an expression more than 1,000 levels deep. No real filter
# comes near this many.
_MAX_FILTER_ITEMS = 100

# Older SQLite, such as 3.34, cuts JSON text at this character, so tags and metadata, kept as
# JSON, refuse it: a filter would otherwise keep the wrong records there.
_NUL = "\x00"

_MODES: tuple[SearchMode, ...] = get_args(SearchMode.__value__)


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


def importance(value: object, *, name: str = "importance") -> float:
    """A real number from 0 to 1, as ``float``. ``bool`` and NaN are rejected."""
    number = _as_float(value)
    if number is None or not 0.0 <= number <= 1.0:  # NaN fails the range check too
        raise ValidationError(f"{name} must be a number from 0 to 1")
    return abs(number)  # SQLite reads -0.0 back as 0.0


def metadata(value: object) -> dict[str, JSONValue]:
    """A deep copy: JSON values under string keys, at most 100 levels deep, with no NUL
    character in any key or text.

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
    """Non-empty strings without a NUL character, without duplicates, in first-seen order."""
    if isinstance(value, str) or not isinstance(value, Iterable):
        raise ValidationError("tags must be a list of strings")
    return tuple(dict.fromkeys(_without_nul("tag", text("tag", tag)) for tag in value))


def tag_filter(value: object) -> tuple[str, ...]:
    """Tags to filter by: what ``tags`` takes, at most 100 of them."""
    result = tags(value)
    if len(result) > _MAX_FILTER_ITEMS:
        raise ValidationError(f"tags must not hold more than {_MAX_FILTER_ITEMS} tags to match")
    return result


def metadata_values(value: object) -> tuple[tuple[str, JSONScalar], ...]:
    """Top-level metadata fields, each with the value it must equal, in the order given.

    At most 100 fields, and no NUL character in a key or text. A value is a string, ``True``,
    ``False``, ``None``, or a number metadata can hold: a finite float or an integer that fits
    in 64 bits.
    """
    if not isinstance(value, Mapping):
        raise ValidationError(f"metadata must be a mapping, not {type(value).__name__}")
    if len(value) > _MAX_FILTER_ITEMS:
        raise ValidationError(
            f"metadata must not hold more than {_MAX_FILTER_ITEMS} fields to match"
        )
    return tuple((_json_text("metadata key", key), _scalar(item)) for key, item in value.items())


def moment(name: str, value: object) -> datetime:
    """A ``datetime`` with a time zone, as the same moment in UTC."""
    cause: Exception | None = None
    try:
        if isinstance(value, datetime) and value.utcoffset() is not None:
            return value.astimezone(UTC)
    except (ValueError, OverflowError) as error:  # a broken time zone, or no such UTC moment
        cause = error
    raise ValidationError(
        f"{name} must be a datetime with a time zone, within years 1 to 9999 in UTC"
    ) from cause


def expiry(value: object) -> datetime | None:
    """Never, or a ``datetime`` with a time zone, as the same moment in UTC."""
    return None if value is None else moment("expires_at", value)


def _shown(value: object) -> str:
    """The start of ``value`` as Python writes it, for a message, or what it is if Python
    refuses to write it, as for an integer of more digits than Python's limit."""
    try:
        return f"{value!r:.40}"
    except ValueError:
        return f"a {type(value).__name__} too long to show"


def flag(name: str, value: object) -> bool:
    """``True`` or ``False``, and nothing that only counts as one, such as ``1``."""
    if isinstance(value, bool):
        return value
    raise ValidationError(f"{name} must be True or False, not {_shown(value)}")


def reject_unknown(method: str, given: Mapping[str, object], known: frozenset[str]) -> None:
    """Raise the ``TypeError`` Python raises for a keyword a function does not take."""
    unknown = sorted(given.keys() - known)
    if unknown:
        raise TypeError(f"{method}() got an unexpected keyword argument {unknown[0]!r}")


def limit(value: object) -> int | None:
    """No limit, or an integer from 1."""
    return None if value is None else _integer("limit", value, minimum=1)


def offset(value: object) -> int:
    """An integer from 0."""
    return _integer("offset", value, minimum=0)


def query(value: object) -> str:
    """Any string to search for."""
    return _string("query", value)


def search_limit(value: object) -> int:
    """How many records a search returns at most: an integer from 1."""
    return _integer("limit", value, minimum=1)


def mode(value: object) -> SearchMode:
    """How to search: ``"auto"``, ``"text"`` or ``"vector"``."""
    for known in _MODES:
        if isinstance(value, str) and value == known:
            return known
    raise ValidationError(f"mode must be 'auto', 'text' or 'vector', not {_shown(value)}")


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
    return _without_nul("path", result)


def embedder(value: object) -> Embedder | None:
    """No model, or an ``Embedder`` with a non-empty name, a callable ``embed`` and a
    ``batch_size`` from 1."""
    if value is None:
        return None
    if not isinstance(value, Embedder):
        raise ValidationError(f"embedder must be an Embedder, not {type(value).__name__}")
    name = text("embedder name", value.name)
    if not callable(value.embed):
        raise ValidationError("embedder embed must be callable")
    size = _integer("batch_size", value.batch_size, minimum=1)
    return Embedder(name, value.embed, batch_size=size)


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


def _json_text(name: str, value: object) -> str:
    """A string to match against text kept as JSON."""
    return _without_nul(name, _string(name, value))


def _without_nul(name: str, value: str) -> str:
    if _NUL in value:
        raise ValidationError(f"{name} must not contain a NUL character")
    return value


def _as_float(value: object) -> float | None:
    """``value`` as a float if it is a real number other than ``bool`` that fits one."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    try:
        return float(value)
    except OverflowError:  # an integer too large for a float
        return None


def _scalar(value: object) -> JSONScalar:
    """A metadata value to match, as a plain ``str``, ``int`` or ``float``, or itself."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return _json_text("metadata value", value)
    if isinstance(value, int) and _fits_64_bits(value):
        return int(value)
    if isinstance(value, float) and math.isfinite(value):
        return float(value)
    raise ValidationError(
        "metadata values to match must be text, True, False, None, a finite float or an"
        f" integer that fits in 64 bits, not {_shown(value)}"
    )


def _fits_64_bits(number: int) -> bool:
    """Whether SQLite can store ``number`` as an integer."""
    return _MIN_INTEGER <= number <= _MAX_INTEGER


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
    if isinstance(value, int) and not _fits_64_bits(value):
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
    """The size of a metadata key or text in UTF-8, if it fits in ``bytes_left`` and holds
    no NUL character."""
    # Every character takes at least a byte, so a text with more characters cannot fit:
    # no need to encode it. Lone surrogates are rejected later, when the whole is encoded.
    too_long = len(text) > bytes_left
    size = len(text) if too_long else len(text.encode("utf-8", "surrogatepass"))
    if size > bytes_left:
        raise ValidationError(f"metadata must not hold more than {_MAX_TEXT_BYTES:,} bytes of text")
    _without_nul("metadata", text)
    return size


def _integer(name: str, value: object, *, minimum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= _MAX_INTEGER
    ):
        raise ValidationError(f"{name} must be an integer from {minimum} to {_MAX_INTEGER}")
    return int(value)
