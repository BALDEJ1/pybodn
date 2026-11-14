"""Which records ``list`` and ``count`` keep: the filters a caller passes, checked once.

``Filter.matches`` is what each filter means. The in-memory store calls it; the SQLite store
asks the same questions in SQL, and the shared tests hold both to the same answers.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from typing import Any, TypedDict

from pybodn import _validation as validate
from pybodn._models import JSONScalar, JSONValue, MemoryRecord


class Filters(TypedDict, total=False):
    """The filters, passed by name. A record is kept only if it passes every one given."""

    tags: Sequence[str]
    metadata: Mapping[str, JSONScalar]
    min_importance: float
    created_since: datetime
    created_before: datetime


@dataclass(frozen=True, slots=True, kw_only=True)
class Filter:
    """Checked filters. The defaults keep every record."""

    tags: tuple[str, ...] = ()
    metadata: tuple[tuple[str, JSONScalar], ...] = ()
    min_importance: float | None = None
    created_since: datetime | None = None  # in UTC, as is created_before
    created_before: datetime | None = None

    def matches(self, record: MemoryRecord) -> bool:
        """Whether ``record`` passes every filter."""
        return (
            all(tag in record.tags for tag in self.tags)
            and all(
                key in record.metadata and _equal(record.metadata[key], value)
                for key, value in self.metadata
            )
            and (self.min_importance is None or record.importance >= self.min_importance)
            and (self.created_since is None or record.created_at >= self.created_since)
            and (self.created_before is None or record.created_at < self.created_before)
        )


EVERYTHING = Filter()

# Each filter, and the check that turns the value given into the value compared.
_CHECKS: dict[str, Callable[[object], Any]] = {
    "tags": validate.tag_filter,
    "metadata": validate.metadata_values,
    "min_importance": partial(validate.importance, name="min_importance"),
    "created_since": partial(validate.moment, "created_since"),
    "created_before": partial(validate.moment, "created_before"),
}


def checked(filters: Mapping[str, object]) -> Filter:
    """``filters`` checked, as a ``Filter``. Rejecting an unknown name is the caller's job."""
    return Filter(
        **{name: check(filters[name]) for name, check in _CHECKS.items() if name in filters}
    )


def _equal(stored: JSONValue, wanted: JSONScalar) -> bool:
    """Equal as JSON values: ``True`` is not ``1``, but ``1`` is ``1.0``."""
    if isinstance(stored, bool) or isinstance(wanted, bool):
        return stored is wanted
    return stored == wanted
