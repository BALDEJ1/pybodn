"""The async engine on each store: every method reaches it, and many calls at once all land."""

import asyncio

from pybodn import AsyncMemoryEngine, NewMemory
from pybodn._storage.base import Store
from tests.contract.conftest import engine_on

CALLS = 50


def test_every_method_reaches_the_store(store: Store) -> None:
    async def use_every_method() -> None:
        async with AsyncMemoryEngine(engine_on(store)) as engine:
            # Act
            notes = engine.collection("user:123")
            coffee = await notes.add("Любить каву", key="coffee", dedupe=True)
            tea, milk = await notes.add_many([NewMemory("Чай"), NewMemory("Молоко")])
            await notes.put("tea", "Чай з молоком")
            changed = await notes.update(coffee.id, content="Любить каву з молоком")
            await notes.archive(tea.id)
            await notes.restore(tea.id)
            deleted = await notes.delete(milk.id)

            # Assert
            assert deleted
            assert await notes.get(coffee.id) == changed
            by_key = await notes.get_by_key("tea")
            assert by_key is not None
            assert by_key.content == "Чай з молоком"
            assert [record.content for record in await notes.list()] == [
                "Любить каву з молоком",
                "Чай",
                "Чай з молоком",
            ]
            assert await notes.count() == 3
            assert [hit.record for hit in await notes.search("кава")] == [changed]
            assert await engine.purge_expired() == 0

    asyncio.run(use_every_method())


def test_many_adds_at_once_all_land(store: Store) -> None:
    async def add_at_once() -> None:
        async with AsyncMemoryEngine(engine_on(store)) as engine:
            # Act
            notes = engine.collection("user:123")
            added = await asyncio.gather(*(notes.add(f"note {number}") for number in range(CALLS)))

            # Assert
            assert len({record.id for record in added}) == CALLS
            assert sorted(record.id for record in await notes.list()) == sorted(
                record.id for record in added
            )

    asyncio.run(add_at_once())
