"""Which records reads keep: the filters a caller passes, and the hidden records they leave out
unless asked, checked once.

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


class Visibility(TypedDict, total=False):
    """Which hidden records a read shows too: archived ones, expired ones, or both.

    ``None`` asks for nothing, as leaving the option out does.
    """

    include_archived: bool | None
    include_expired: bool | None


class Filters(Visibility, total=False):
    """The filters, passed by name. A record is kept only if it passes every one given.

    ``None`` sets no condition, as leaving the filter out does.
    """

    tags: Sequence[str] | None
    metadata: Mapping[str, JSONScalar] | None
    min_importance: float | None
    created_since: datetime | None
    created_before: datetime | None


@dataclass(frozen=True, slots=True, kw_only=True)
class Filter:
    """Checked filters, times in UTC. The defaults keep every record, hidden ones too.

    ``hide_archived`` leaves archived records out, and ``hide_expired_at`` the records that have
    expired at that moment.
    """

    tags: tuple[str, ...] = ()
    metadata: tuple[tuple[str, JSONScalar], ...] = ()
    min_importance: float | None = None
    created_since: datetime | None = None
    created_before: datetime | None = None
    hide_archived: bool = False
    hide_expired_at: datetime | None = None

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
            and (not self.hide_archived or record.archived_at is None)
            and (self.hide_expired_at is None or not is_expired(record, self.hide_expired_at))
        )


EVERYTHING = Filter()

# Each filter that sets a condition, and the check that turns the value given into the value
# compared. The options of ``Visibility`` are checked apart: they leave records out by default.
_CHECKS: dict[str, Callable[[object], Any]] = {
    "tags": validate.tag_filter,
    "metadata": validate.metadata_values,
    "min_importance": partial(validate.importance, name="min_importance"),
    "created_since": partial(validate.moment, "created_since"),
    "created_before": partial(validate.moment, "created_before"),
}


def is_expired(record: MemoryRecord, now: datetime) -> bool:
    """Whether ``record`` has expired at ``now``: its expiry is no later than that."""
    return record.expires_at is not None and record.expires_at <= now


def checked(
    method: str,
    filters: Mapping[str, object],
    *,
    now: datetime,
    known: frozenset[str] = Filters.__optional_keys__,
) -> Filter:
    """``filters``, as passed to ``method``, checked into a ``Filter`` that also leaves out the
    records hidden at ``now``, unless the options ask for them.

    ``known`` are the names ``method`` takes: any other raises ``TypeError``. A filter set to
    ``None`` is left out.
    """
    validate.reject_unknown(method, filters, known)
    given = {name: value for name, value in filters.items() if value is not None}
    include_archived = validate.flag("include_archived", given.get("include_archived", False))
    include_expired = validate.flag("include_expired", given.get("include_expired", False))
    return Filter(
        **{name: check(given[name]) for name, check in _CHECKS.items() if name in given},
        hide_archived=not include_archived,
        hide_expired_at=None if include_expired else now,
    )


def _equal(stored: JSONValue, wanted: JSONScalar) -> bool:
    """Equal as JSON values: ``True`` is not ``1``, but ``1`` is ``1.0``."""
    if isinstance(stored, bool) or isinstance(wanted, bool):
        return stored is wanted
    return stored == wanted
