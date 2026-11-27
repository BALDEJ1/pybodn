import inspect
from collections.abc import Callable

import pytest

import pybodn
from pybodn import Collection, Embedder, MemoryEngine

PUBLIC_NAMES = [
    "AsyncCollection",
    "AsyncMemoryEngine",
    "Collection",
    "Embedder",
    "JSONValue",
    "MemoryEngine",
    "MemoryRecord",
    "NewMemory",
    "SearchHit",
    "__version__",
]


def test_the_package_exports_exactly_the_documented_names() -> None:
    public = {
        name
        for name, value in vars(pybodn).items()
        if not name.startswith("_") and not inspect.ismodule(value)
    }
    assert sorted(pybodn.__all__) == sorted(PUBLIC_NAMES)
    # __version__ is in __all__, though its name starts with an underscore.
    assert public == set(pybodn.__all__) - {"__version__"}


@pytest.mark.parametrize("name", PUBLIC_NAMES)
def test_every_documented_name_can_be_imported(name: str) -> None:
    assert hasattr(pybodn, name)


# New parameters stay keyword-only, so adding them later breaks no caller.
@pytest.mark.parametrize(
    ("method", "positional"),
    [
        (MemoryEngine.collection, ["namespace"]),
        (MemoryEngine.embed_missing, []),
        (MemoryEngine.purge_expired, []),
        (MemoryEngine.erase_deleted, []),
        (MemoryEngine.namespaces, []),
        (Collection.add, ["content"]),
        (Collection.add_many, ["memories"]),
        (Collection.put, ["key", "content"]),
        (Collection.get, ["id"]),
        (Collection.get_by_key, ["key"]),
        (Collection.update, ["id"]),
        (Collection.delete, ["id"]),
        (Collection.archive, ["id"]),
        (Collection.restore, ["id"]),
        (Collection.list, []),
        (Collection.count, []),
        (Collection.search, ["query"]),
    ],
)
def test_only_required_parameters_can_be_passed_by_position(
    method: Callable[..., object], positional: list[str]
) -> None:
    # Arrange
    parameters = list(inspect.signature(method).parameters.values())[1:]  # without self

    # Act
    by_position = [
        parameter.name
        for parameter in parameters
        if parameter.kind not in (parameter.KEYWORD_ONLY, parameter.VAR_KEYWORD)
    ]

    # Assert
    assert by_position == positional


@pytest.mark.parametrize(
    ("factory", "positional"),
    [(MemoryEngine.in_memory, []), (MemoryEngine.sqlite, ["path"])],
)
def test_engine_factories_take_only_required_parameters_by_position(
    factory: Callable[..., object], positional: list[str]
) -> None:
    # Arrange: the class is already bound, so every parameter is the caller's
    parameters = inspect.signature(factory).parameters.values()

    # Act
    by_position = [
        parameter.name for parameter in parameters if parameter.kind != parameter.KEYWORD_ONLY
    ]

    # Assert
    assert by_position == positional


def test_an_embedder_takes_only_its_name_and_function_by_position() -> None:
    # Arrange
    parameters = inspect.signature(Embedder).parameters.values()

    # Act
    by_position = [
        parameter.name for parameter in parameters if parameter.kind != parameter.KEYWORD_ONLY
    ]

    # Assert
    assert by_position == ["name", "embed"]
