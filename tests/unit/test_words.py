"""How text becomes words: the same rules for stored records and for search queries."""

import re
import sys
import unicodedata

import pytest

from pybodn._words import WORD_RULES, words

EXAMPLES = [
    pytest.param("Hello, World!", ["hello", "world"], id="case-and-punctuation"),
    pytest.param("STRASSE straße", ["strasse", "strasse"], id="full-case-folding"),
    pytest.param("ＡＢＣ１２３ ﬁle x²", ["abc123", "file", "x2"], id="compatibility-forms"),
    pytest.param("Ёлка е\u0308лка", ["елка", "елка"], id="yo-is-ye"),
    pytest.param("алергі\u0301я до\u0300м", ["алергія", "дом"], id="stress-marks-dropped"),
    pytest.param(
        "і\u0301\u0308жак е\u0301\u0308лка краи\u0301\u0306",
        ["їжак", "елка", "край"],
        id="letters-whole-when-a-stress-mark-sits-in-between",
    ),
    pytest.param("café cafe\u0301", ["café", "café"], id="latin-accents-kept"),
    pytest.param("п'є п\u2019є п\u02bcє п`є", ["пє", "пє", "пє", "пє"], id="apostrophes-dropped"),
    pytest.param("їжак й ґанок і\u0308жак", ["їжак", "й", "ґанок", "їжак"], id="ukrainian-letters"),
    pytest.param("हिन्दी", ["हिन्दी"], id="marks-stay-in-the-word"),
    pytest.param("我喜欢咖啡", ["我喜欢咖啡"], id="no-spaces-is-one-word"),
    pytest.param("v2.0, 3-й user_id", ["v2", "0", "3", "й", "user", "id"], id="separators"),
    pytest.param("Thanks🙏 tea\u00a0milk\u3000rice", ["thanks", "tea", "milk", "rice"], id="emoji"),
    pytest.param(
        "\U00020000\U00020001 \U0001e900\U0001e901",
        ["\U00020000\U00020001", "\U0001e922\U0001e923"],
        id="beyond-the-basic-plane",
    ),
    pytest.param("кава кава", ["кава", "кава"], id="repeats-kept"),
    pytest.param("", [], id="empty"),
    pytest.param("??? 🙏 ☕ ' \u2019 \u0301", [], id="no-letters-or-numbers"),
]


@pytest.mark.parametrize(("text", "expected"), EXAMPLES)
def test_text_becomes_these_words(text: str, expected: list[str]) -> None:
    assert words(text) == expected


def test_a_word_is_a_run_of_letters_numbers_and_marks() -> None:
    # Arrange: every character there is, and the rules written the slow, obvious way
    text = "".join(map(chr, range(sys.maxunicode + 1)))
    folded = unicodedata.normalize("NFKC", text).casefold()
    kept = "".join(
        character for character in folded if character not in "\u0300\u0301'\u2019\u02bc`"
    )
    plain = unicodedata.normalize("NFKC", kept).replace("ё", "е")
    spaced = "".join(
        character if unicodedata.category(character)[0] in "LNM" else " " for character in plain
    )

    # Act
    result = words(text)

    # Assert: the same words, and each one is a single token for SQLite's ascii tokenizer
    assert result == spaced.split()
    assert all(re.fullmatch("[a-z0-9\x80-\U0010ffff]+", word) for word in result)


def test_changing_the_rules_needs_a_new_rules_number() -> None:
    # Files remember the number their word index was built with and rebuild it when this grows.
    # Once these rules are released, a change to any test above needs a new number here and in
    # WORD_RULES.
    assert WORD_RULES == 1
