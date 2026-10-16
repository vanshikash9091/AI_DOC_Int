
import re


def normalize_text(text: str) -> str:
    """
    Clean unnecessary whitespace from extracted document text.
    """
    text = text.replace("\x00", "")

    # Replace repeated spaces and tabs with one space
    text = re.sub(r"[ \t]+", " ", text)

    # Replace three or more newlines with two newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove whitespace at the beginning and end
    return text.strip()


def chunk_text(
    text: str,
    max_words: int = 400,
    overlap_words: int = 50
) -> list[str]:
    """
    Split text into word-based chunks with overlapping context.
    """
    if max_words < 1:
        raise ValueError("max_words must be at least 1")

    if overlap_words < 0 or overlap_words >= max_words:
        raise ValueError(
            "overlap_words must be non-negative and less than max_words"
        )

    words = text.split()

    if not words:
        return []

    chunks = []
    step = max_words - overlap_words

    for start in range(0, len(words), step):
        end = start + max_words
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        if end >= len(words):
            break

    return chunks
