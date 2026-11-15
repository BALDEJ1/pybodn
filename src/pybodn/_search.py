"""Search by words, the same for every store: the terms a query looks for, and how records score.

A query word looks for itself and for its other forms: any word that starts as it does once its
ending is cut off, so "алергії" also finds "алергія". A long word with no known ending finds the
words that start with it, so "лікар" finds "лікаря". A short word or a number finds only itself,
so that "про" does not find every word that starts with "про", nor "1000" find "10000". Forms
that change inside the word, such as "рік" and "року" or "run" and "running", are not found. A
start of three letters also finds some other words: "this" finds "thing", and "його" finds
"йога". Such a record matches only the start, so it usually ranks below records that hold the
query's own words.

Records score as SQLite's full-text search scores them: BM25, step by step as FTS5's ``bm25()``
computes it, over the words of every record in the store, rounded to ``SCORE_DECIMALS`` places.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

from pybodn._words import words
from pybodn.errors import ValidationError

# Each word is a term or two of one search, and a long query is slow: 100 words take 0.3 s on
# 100,000 records, 1,000 words 3 s. A word repeated in the query counts once.
MAX_QUERY_WORDS = 100

# The fewest letters an ending may leave: a shorter start would find too many unrelated words.
_MIN_STEM = 3

# A word with no known ending, at least this long, also finds the words that start with it.
_LONG_WORD = 4

# Endings a query word may end in, cut off to reach the start that its other forms share; the
# longest that fits is cut. Ukrainian and Russian: nouns and adjectives in every case, verbs in
# the infinitive, the third person and the masculine past. English: plurals and verb forms.
# Endings that also end many nouns are left out: "-ет" would cut "інтернет" to "інтерн", and
# "-ла" "школа" to "шко". tests/unit/test_word_forms.py holds a pair of real words that needs
# each ending.
_ENDINGS = sorted(
    (
        # nouns and adjectives
        "а я о е є и і ї ы у ю ь ам ям ом ем єм им ім ым ах ях их іх ых ою ею єю ую юю ов ев ів"
        " їв ой ей ий ій ый ая яя ое ее ие ые ої ами ями ими іми ыми ого его ому ему ові еві єві"
        " ього ьому ьої ьою"
        # verbs; "єть" is what "-ється" leaves once the reflexive particle is cut
        " ти ть ати яти ити іти ать ять ить еть уть ють єть в вати овать евать"
        # English
        " s es ies ed ied ing"
    ).split(),
    key=len,
    reverse=True,
)

# The particle of reflexive verbs, cut before the ending: "вчився" -> "вчив" -> "вчи".
_REFLEXIVE = ("ся", "сь")

# FTS5's bm25() constants: how soon a repeated word stops adding to the score, and how much a
# long record is held back. A word in over half of all records weighs this little, not less.
_K1 = 1.2
_B = 0.75
_MIN_WEIGHT = 1e-6

# SQLite's word index holds each record's namespace too, as one more token. It weighs nothing,
# but it makes every record one token longer, so the lengths here count it as well.
_NAMESPACE_TOKENS = 1

# Scores are rounded to this many decimal places, in both stores. A processor that fuses each
# multiplication with the addition after it, as Apple's and other ARM chips do, leaves SQLite's
# sums a last digit apart from Python's, and records that score alike would change places.
SCORE_DECIMALS = 9


@dataclass(frozen=True, slots=True)
class Term:
    """What a query looks for: the word ``text``, or with ``is_prefix`` any word starting so."""

    text: str
    is_prefix: bool

    def count_in(self, record_words: Sequence[str]) -> int:
        """How many of ``record_words`` this term finds."""
        if self.is_prefix:
            return sum(word.startswith(self.text) for word in record_words)
        return record_words.count(self.text)


def query_terms(query: str) -> tuple[Term, ...]:
    """The terms ``query`` looks for, each once, in the order of its words.

    A query of more than ``MAX_QUERY_WORDS`` different words raises ``ValidationError``.
    """
    distinct = dict.fromkeys(words(query))
    if len(distinct) > MAX_QUERY_WORDS:
        raise ValidationError(
            f"query must not hold more than {MAX_QUERY_WORDS} different words; this one has"
            f" {len(distinct)}"
        )
    terms: dict[Term, None] = {}
    for word in distinct:
        terms[Term(word, is_prefix=False)] = None
        start = _start(word)
        if start is not None:
            terms[Term(start, is_prefix=True)] = None
    return tuple(terms)


def bm25_scores(terms: Sequence[Term], every_record: Sequence[Sequence[str]]) -> list[float | None]:
    """How well each record's words match ``terms``: higher is better, ``None`` for no match.

    ``every_record`` holds the words of every record in the store, all namespaces included, as
    FTS5 counts its whole table. The arithmetic follows FTS5's, so the scores agree with SQLite's.
    """
    if not every_record:
        return []
    rows = len(every_record)
    lengths = [len(record) + _NAMESPACE_TOKENS for record in every_record]
    average_length = sum(lengths) / rows
    counts = [[term.count_in(record) for term in terms] for record in every_record]
    weights = [_weight(rows, sum(1 for found in counts if found[i])) for i in range(len(terms))]
    return [
        _score(found, length, average_length, weights) if any(found) else None
        for found, length in zip(counts, lengths, strict=True)
    ]


def _start(word: str) -> str | None:
    """The start that other forms of ``word`` share, or ``None`` if it finds only itself."""
    if word.isdigit():  # a number has no other forms: "1000" must not find "10000"
        return None
    base = _cut(word, _REFLEXIVE) or word
    return _cut(base, _ENDINGS) or (base if len(word) >= _LONG_WORD else None)


def _cut(word: str, endings: Sequence[str]) -> str | None:
    """``word`` without the first of ``endings`` it ends in, if enough is left; else ``None``."""
    for ending in endings:
        if word.endswith(ending) and len(word) - len(ending) >= _MIN_STEM:
            return word[: -len(ending)]
    return None


def _weight(rows: int, rows_found: int) -> float:
    """How much a term counts: less the more records it finds (FTS5's IDF)."""
    weight = math.log((rows - rows_found + 0.5) / (rows_found + 0.5))
    return weight if weight > 0.0 else _MIN_WEIGHT


def _score(
    found: Sequence[int], length: int, average_length: float, weights: Sequence[float]
) -> float:
    """One record's score: each term in turn, in FTS5's order of operations, then rounded."""
    score = 0.0
    for count, weight in zip(found, weights, strict=True):
        score += weight * (
            (count * (_K1 + 1.0)) / (count + _K1 * (1 - _B + _B * length / average_length))
        )
    return round(score, SCORE_DECIMALS)
