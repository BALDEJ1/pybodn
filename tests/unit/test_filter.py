from dataclasses import fields

import pytest

from pybodn._filter import _CHECKS, EVERYTHING, Filter, Filters, checked


def test_every_filter_has_exactly_one_check_and_one_field() -> None:
    assert set(_CHECKS) == Filters.__optional_keys__
    assert set(_CHECKS) == {field.name for field in fields(Filter)}


def test_no_filters_and_filters_set_to_none_keep_everything() -> None:
    assert checked("list", {}) == EVERYTHING
    assert checked("list", dict.fromkeys(_CHECKS)) == EVERYTHING


def test_an_unknown_filter_is_a_type_error_naming_the_method() -> None:
    with pytest.raises(TypeError, match=r"^search\(\) got an unexpected keyword argument 'tag'$"):
        checked("search", {"tag": ["health"]})
