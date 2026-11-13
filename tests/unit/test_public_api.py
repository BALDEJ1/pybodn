import inspect
from collections.abc import Callable

import pytest

import pybodn
from pybodn import Collection, MemoryEngine

PUBLIC_NAMES = [
    "Collection",
    "JSONValue",
    "MemoryEngine",
    "MemoryRecord",
    "NewMemory",
    "__version__",
]


def test_the_package_exports_exactly_the_documented_names() -> None:
    assert sorted(pybodn.__all__) == sorted(PUBLIC_NAMES)


@pytest.mark.parametrize("name", PUBLIC_NAMES)
def test_every_documented_name_can_be_imported(name: str) -> None:
    assert hasattr(pybodn, name)


# New parameters stay keyword-only, so adding them later breaks no caller.
@pytest.mark.parametrize(
    ("method", "positional"),
    [
        (MemoryEngine.collection, ["namespace"]),
        (Collection.add, ["content"]),
        (Collection.add_many, ["memories"]),
        (Collection.put, ["key", "content"]),
        (Collection.get, ["id"]),
        (Collection.get_by_key, ["key"]),
        (Collection.update, ["id"]),
        (Collection.delete, ["id"]),
        (Collection.list, []),
        (Collection.count, []),
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
