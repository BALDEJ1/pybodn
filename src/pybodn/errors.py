"""Errors raised by pybodn. Catch ``PyBodnError`` to handle any of them."""

__all__ = [
    "ConcurrentUpdateError",
    "DuplicateKeyError",
    "EngineClosedError",
    "PyBodnError",
    "RecordNotFoundError",
    "StorageError",
    "ValidationError",
]


class PyBodnError(Exception):
    """Base class for every error raised by pybodn."""


class ValidationError(PyBodnError, ValueError):
    """Input data is invalid. The message names the field."""


class RecordNotFoundError(PyBodnError, LookupError):
    """The record to update does not exist."""


class DuplicateKeyError(PyBodnError):
    """The key is already used in the namespace, or repeats within one batch."""


class ConcurrentUpdateError(PyBodnError):
    """The record changed after the version passed as ``expected_version``."""


class StorageError(PyBodnError):
    """The storage failed. The original error is the cause."""


class EngineClosedError(PyBodnError):
    """The engine was used after ``close()``."""
