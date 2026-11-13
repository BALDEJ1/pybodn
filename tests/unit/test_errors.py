import pytest

from pybodn import errors


@pytest.mark.parametrize("name", errors.__all__)
def test_every_error_is_a_pybodn_error(name: str) -> None:
    assert issubclass(getattr(errors, name), errors.PyBodnError)


def test_validation_error_is_a_value_error() -> None:
    assert issubclass(errors.ValidationError, ValueError)


def test_record_not_found_error_is_a_lookup_error() -> None:
    assert issubclass(errors.RecordNotFoundError, LookupError)
