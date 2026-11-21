"""Errors raised by pybodn. Catch ``PyBodnError`` to handle any of them."""

__all__ = [
    "ConcurrentUpdateError",
    "ConfigurationError",
    "DuplicateKeyError",
    "EmbeddingError",
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
    """The record to update, archive or restore does not exist."""


class DuplicateKeyError(PyBodnError):
    """The key is already used in the namespace, or repeats within one batch."""


class ConcurrentUpdateError(PyBodnError):
    """The record changed after the version passed as ``expected_version``."""


class StorageError(PyBodnError):
    """The storage failed. The original error, if there is one, is its cause."""


class EngineClosedError(PyBodnError):
    """The engine was used after ``close()``."""


class ConfigurationError(PyBodnError):
    """The engine is not set up for what was asked, such as a search by meaning without a model."""


class EmbeddingError(PyBodnError):
    """The embedding model gave something unusable: not one vector of numbers per text, a vector
    that cannot be scaled to unit length, or one of another length than its stored vectors."""
