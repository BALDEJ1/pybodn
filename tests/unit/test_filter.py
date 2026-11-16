from dataclasses import fields, replace
from datetime import UTC, datetime, timedelta

import pytest

from pybodn._filter import _CHECKS, EVERYTHING, Filter, Filters, Visibility, checked, is_expired
from tests.unit.records import RECORD

NOW = datetime(2026, 11, 16, 12, 0, tzinfo=UTC)
VISIBLE_NOW = Filter(hide_archived=True, hide_expired_at=NOW)


def test_every_filter_has_exactly_one_check_and_one_field() -> None:
    assert set(_CHECKS) == Filters.__optional_keys__ - Visibility.__optional_keys__
    assert {field.name for field in fields(Filter)} == {
        *_CHECKS,
        "hide_archived",
        "hide_expired_at",
    }


def test_no_filters_and_filters_set_to_none_leave_out_only_the_records_hidden_now() -> None:
    assert checked("list", {}, now=NOW) == VISIBLE_NOW
    assert checked("list", dict.fromkeys(Filters.__optional_keys__), now=NOW) == VISIBLE_NOW


def test_both_options_keep_everything() -> None:
    options = {"include_archived": True, "include_expired": True}
    assert checked("list", options, now=NOW) == EVERYTHING


def test_the_default_filter_keeps_hidden_records_too() -> None:
    hidden = replace(RECORD, archived_at=NOW, expires_at=NOW)
    assert EVERYTHING.matches(hidden)
    assert not VISIBLE_NOW.matches(hidden)


def test_a_record_expires_at_its_expiry_and_never_without_one() -> None:
    record = replace(RECORD, expires_at=NOW)
    assert not is_expired(record, NOW - timedelta(microseconds=1))
    assert is_expired(record, NOW)
    assert not is_expired(RECORD, NOW)


def test_an_unknown_filter_is_a_type_error_naming_the_method() -> None:
    with pytest.raises(TypeError, match=r"^search\(\) got an unexpected keyword argument 'tag'$"):
        checked("search", {"tag": ["health"]}, now=NOW)
