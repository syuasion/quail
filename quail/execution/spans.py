"""Reading a copied answer from decoded text and locating it in the document."""

import re


def read_phrase(decoded: str) -> str | None:
    """Return the words the model wrote after the phrase cue.

    The answer is the whole text, up to where the model ended its turn.

    Args:
        decoded: The text the model wrote after the phrase cue.

    Returns:
        The answer, or None for an empty answer.
    """
    return decoded.strip() or None


def locate_span(document: str, phrase: str) -> tuple[int, int] | None:
    """Return the character span of the phrase's first occurrence in the document.

    An exact match wins. Otherwise case is ignored and any run of
    whitespace in the phrase matches any run of whitespace in the
    document.

    Returns:
        (start, end), or None when the document does not hold the phrase.
    """
    start = document.find(phrase)
    if start >= 0:
        return start, start + len(phrase)
    words = phrase.split()
    if not words:
        return None
    pattern = r"\s+".join(re.escape(word) for word in words)
    match = re.search(pattern, document, re.IGNORECASE)
    return None if match is None else match.span()
