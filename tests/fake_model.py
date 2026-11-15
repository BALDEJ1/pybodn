"""An embedding model for the tests: predictable vectors, no network, and a record of each call."""

# Each word here is one number of a vector: how often the text holds it. Every vector starts with
# a one, so that a text without these words still has a vector.
WORDS = ("кава", "чай", "молоко", "арахіс", "алергія", "python")


def vector_of(text: str) -> list[float]:
    """The vector this model gives ``text``."""
    text_words = text.lower().split()
    return [1.0, *(float(text_words.count(word)) for word in WORDS)]


def broken(texts: list[str]) -> list[list[float]]:
    """A model that is down."""
    raise RuntimeError("the model is down")


class FakeModel:
    """Gives each text its ``vector_of``, and keeps the texts of every call in ``calls``."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [vector_of(text) for text in texts]
