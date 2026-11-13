"""Exact-match hash index."""

from collections.abc import Hashable


class HashIndex[K: Hashable, V]:
    """Exact-match lookup from a hashable key to a value.

    Backed by a ``dict``, so lookups are average-case O(1), not worst-case.
    An index is derived, rebuildable state; it is never the durable copy of data.

    ``get`` returns ``None`` for a missing key, so storing ``None`` as a value
    is indistinguishable from absence; use ``in`` when that matters.
    """

    __slots__ = ("_entries",)

    def __init__(self) -> None:
        self._entries: dict[K, V] = {}

    def set(self, key: K, value: V) -> None:
        """Map ``key`` to ``value``, replacing any previous value."""
        self._entries[key] = value

    def get(self, key: K) -> V | None:
        """Return the value for ``key``, or ``None`` if it is absent."""
        return self._entries.get(key)

    def delete(self, key: K) -> bool:
        """Remove ``key``; return ``True`` if it was present. Safe to repeat."""
        if key not in self._entries:
            return False
        del self._entries[key]
        return True

    def clear(self) -> None:
        """Remove every key."""
        self._entries.clear()

    def __contains__(self, key: object) -> bool:
        return key in self._entries

    def __len__(self) -> int:
        return len(self._entries)
