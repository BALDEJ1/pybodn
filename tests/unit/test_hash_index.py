from pybodn.indexes import HashIndex


def test_get_returns_value_after_set() -> None:
    index: HashIndex[str, str] = HashIndex()

    index.set("user:123:language", "record_001")

    assert index.get("user:123:language") == "record_001"


def test_get_returns_none_for_missing_key() -> None:
    index: HashIndex[str, str] = HashIndex()

    assert index.get("missing") is None


def test_set_replaces_existing_value() -> None:
    index: HashIndex[str, str] = HashIndex()
    index.set("key", "old")

    index.set("key", "new")

    assert index.get("key") == "new"
    assert len(index) == 1


def test_delete_removes_key_and_returns_true() -> None:
    index: HashIndex[str, str] = HashIndex()
    index.set("key", "value")

    removed = index.delete("key")

    assert removed is True
    assert "key" not in index
    assert index.get("key") is None


def test_delete_missing_key_returns_false() -> None:
    index: HashIndex[str, str] = HashIndex()

    assert index.delete("missing") is False


def test_delete_twice_is_safe() -> None:
    index: HashIndex[str, str] = HashIndex()
    index.set("key", "value")

    first = index.delete("key")
    second = index.delete("key")

    assert (first, second) == (True, False)


def test_contains_reports_membership() -> None:
    index: HashIndex[str, str] = HashIndex()
    index.set("present", "value")

    assert "present" in index
    assert "absent" not in index


def test_len_counts_keys() -> None:
    index: HashIndex[str, int] = HashIndex()

    index.set("a", 1)
    index.set("b", 2)

    assert len(index) == 2


def test_clear_removes_all_keys() -> None:
    index: HashIndex[str, int] = HashIndex()
    index.set("a", 1)
    index.set("b", 2)

    index.clear()

    assert len(index) == 0
    assert index.get("a") is None


def test_supports_namespace_key_tuples() -> None:
    index: HashIndex[tuple[str, str], str] = HashIndex()

    index.set(("user:123", "language"), "record_001")

    assert index.get(("user:123", "language")) == "record_001"
    assert index.get(("user:456", "language")) is None


def test_instances_do_not_share_state() -> None:
    first: HashIndex[str, str] = HashIndex()
    second: HashIndex[str, str] = HashIndex()

    first.set("key", "value")

    assert "key" not in second
    assert len(second) == 0
