"""How text becomes words: the same rules for stored records and for search queries."""

import re
import sys
import unicodedata

import pytest

from pybodn._words import words

# Files remember WORD_RULES, the number their word index was built with, and rebuild it when the
# number grows. Once these rules are released, changing any expected words here needs a new one.
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
    pytest.param("İstanbul ISTANBUL", ["istanbul", "istanbul"], id="turkish-dotted-capital-i"),
    pytest.param("п'є п\u2019є п\u02bcє п`є", ["пє", "пє", "пє", "пє"], id="apostrophes-dropped"),
    pytest.param("їжак й ґанок і\u0308жак", ["їжак", "й", "ґанок", "їжак"], id="ukrainian-letters"),
    pytest.param("हिन्दी", ["हिन्दी"], id="marks-stay-in-the-word"),
    pytest.param("我喜欢咖啡", ["我喜欢咖啡"], id="no-spaces-is-one-word"),
    pytest.param("v2.0, 3-й user_id", ["v2", "0", "3", "й", "user", "id"], id="separators"),
    pytest.param(
        "co\u00adoperate a\u200db \ufeffhello wo\u200erld",
        ["cooperate", "ab", "hello", "world"],
        id="invisible-formatting-dropped",
    ),
    pytest.param("кава\u200bчай", ["кава", "чай"], id="zero-width-space-separates"),
    pytest.param("Thanks🙏 tea\u00a0milk\u3000rice", ["thanks", "tea", "milk", "rice"], id="emoji"),
    pytest.param(
        "кава ❤\ufe0f 葛\U000e0100城 \u1820\u180b\u1822",
        ["кава", "葛城", "\u1820\u1822"],
        id="variation-selectors-dropped",
    ),
    pytest.param(
        "\U00020000\U00020001 \U0001e900\U0001e901",
        ["\U00020000\U00020001", "\U0001e922\U0001e923"],
        id="beyond-the-basic-plane",
    ),
    pytest.param("кава кава", ["кава", "кава"], id="repeats-kept"),
    pytest.param("", [], id="empty"),
    pytest.param("??? 🙏 ☕ ' \u2019 \u0301 \ufe0f", [], id="no-letters-or-numbers"),
]


@pytest.mark.parametrize(("text", "expected"), EXAMPLES)
def test_text_becomes_these_words(text: str, expected: list[str]) -> None:
    assert words(text) == expected


def test_every_word_is_letters_numbers_and_marks_in_one_token() -> None:
    # Act: every character there is, in one text
    result = words("".join(map(chr, range(sys.maxunicode + 1))))

    # Assert: a word holds only letters, numbers and marks, and is a single token for SQLite's
    # ascii tokenizer, which splits at every ASCII character that is not a letter or a digit
    assert all(unicodedata.category(character)[0] in "LNM" for word in result for character in word)
    assert all(re.fullmatch("[a-z0-9\x80-\U0010ffff]+", word) for word in result)
