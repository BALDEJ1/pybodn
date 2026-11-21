"""How text becomes words, the same way for stored records and for search queries.

The text is normalised (NFKC) and case-folded, and "ё" is written as "е". Stress marks,
apostrophes and invisible characters such as a soft hyphen are dropped, so "алергі́я" is
"алергія" and "п’є", "п'є" and "пє" are one word. A word is then a run of letters, numbers and
the marks written on them; everything else separates words, the zero-width space included.
"""

import unicodedata
from functools import lru_cache

# A file stores the number its word index was built with, and rebuilds the index on open when
# that number is older than this one. The number is frozen until 2.0, and so are the words that
# words() gives: every 1.x must search the index any other 1.x built. What counts as a letter
# comes from this Python's Unicode data, so a character that only a newer Unicode knows may
# split differently under an older Python.
WORD_RULES = 1

# Dropped wherever they are, so the letters around them join up: stress marks, which NFKC
# leaves after a Cyrillic letter; the dot that case folding leaves after the "i" of the Turkish
# "İ"; and apostrophes.
_DROPPED = frozenset("\u0300\u0301\u0307'\u2019\u02bc`")
# The one invisible formatting character that is not dropped: it separates words in Thai and
# similar scripts.
_ZERO_WIDTH_SPACE = "\u200b"
# How many characters _part remembers: far more than real text uses, and a bound on memory.
_REMEMBERED_CHARACTERS = 1 << 16


def words(text: str) -> list[str]:
    """The words of ``text`` in order, repeats included."""
    folded = unicodedata.normalize("NFKC", text).casefold()
    # NFKC again: with a stress mark gone from between them, "и" and a breve can become "й".
    plain = unicodedata.normalize("NFKC", "".join(map(_part, folded)))
    return plain.replace("ё", "е").split()


@lru_cache(maxsize=_REMEMBERED_CHARACTERS)
def _part(character: str) -> str:
    """Itself inside a word, nothing when dropped, a space between words."""
    category = unicodedata.category(character)
    if (
        character in _DROPPED
        # invisible formatting, such as a soft hyphen or a zero-width joiner
        or (category == "Cf" and character != _ZERO_WIDTH_SPACE)
        # such as the one after many emoji: it only picks how the character before it looks
        or "VARIATION SELECTOR" in unicodedata.name(character, "")
    ):
        return ""
    return character if category[0] in "LNM" else " "  # a letter, number or mark
