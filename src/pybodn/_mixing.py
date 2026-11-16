"""Mixed search, the same for every store: the records that the search by words and the search by
meaning found, ranked as one by reciprocal rank fusion."""

from collections.abc import Sequence
from fractions import Fraction

from pybodn._models import MemoryRecord, SearchHit

# The larger it is, the less the first places outweigh the rest. Reciprocal rank fusion was
# introduced with 60.
_K = 60


def mix(
    by_words: Sequence[SearchHit], by_meaning: Sequence[SearchHit], *, limit: int
) -> list[SearchHit]:
    """The ``limit`` best records of the two searches, best first.

    A record scores ``1 / (_K + place)`` for its place in each search that found it, counted from
    1, added up. Equal scores keep the order of the search by words, and its records come before
    those only the search by meaning found. A record both found is as the search by words
    returned it.
    """
    # Added up exactly: sums that are equal can differ in their last bit as floats.
    scores: dict[str, Fraction] = {}
    records: dict[str, MemoryRecord] = {}
    for hits in (by_words, by_meaning):
        for place, hit in enumerate(hits, start=1):
            scores[hit.record.id] = scores.get(hit.record.id, 0) + Fraction(1, _K + place)
            records.setdefault(hit.record.id, hit.record)
    best = sorted(scores, key=scores.__getitem__, reverse=True)  # stable: ties keep the order found
    return [
        SearchHit(record=records[record_id], score=float(scores[record_id]))
        for record_id in best[:limit]
    ]
