"""How text becomes words, the same way for stored records and for search queries.

The text is normalised (NFKC) and case-folded. Stress marks and apostrophes are dropped, so
"алергі́я" is "алергія" and "п’є", "п'є" and "пє" are one word, and "ё" is written as "е".
A word is then a run of letters, numbers and the marks written on them; everything else
separates words.
"""

import re
import unicodedata
from functools import cache

# A file stores the number its word index was built with, and rebuilds the index on open when
# that number is older than this one. Raise it whenever words() starts giving different words.
# What counts as a letter comes from this Python's Unicode data, so a character that only a
# newer Unicode knows may split differently under an older Python.
WORD_RULES = 1

# Stress marks, which NFKC leaves after a Cyrillic letter, and apostrophes.
_DROPPED = re.compile("[\u0300\u0301'\u2019\u02bc`]")
# Characters beyond the Basic Multilingual Plane: emoji and rarer scripts.
_ASTRAL = re.compile("[\U00010000-\U0010ffff]")
_BASIC_PLANE_END = 0x10000


def words(text: str) -> list[str]:
    """The words of ``text`` in order, repeats included."""
    folded = unicodedata.normalize("NFKC", text).casefold()
    # NFKC again: with a stress mark gone from between them, "и" and a breve can become "й".
    plain = unicodedata.normalize("NFKC", _DROPPED.sub("", folded)).replace("ё", "е")
    return _word_pattern().findall(_ASTRAL.sub(_astral, plain))


def _in_word(character: str) -> bool:
    return unicodedata.category(character)[0] in "LNM"  # a letter, number or mark


def _astral(match: re.Match[str]) -> str:
    """Keeps an astral character that belongs in a word and turns any other into a space."""
    return match[0] if _in_word(match[0]) else " "


@cache
def _word_pattern() -> re.Pattern[str]:
    """A run of word characters, built on first use from this Python's Unicode data.

    It lists the word characters of the Basic Multilingual Plane, which the regex engine checks
    in one step, and lets every astral character through: ``_astral`` has already turned those
    outside a word into spaces. Listing astral characters too would make every space slow.
    """
    basic = "".join(filter(_in_word, map(chr, range(_BASIC_PLANE_END))))
    return re.compile(f"[{basic}\U00010000-\U0010ffff]+")  # no word character needs escaping
