
from functools import lru_cache

from transformers import pipeline

from app.text_processor import chunk_text


MODEL_NAME = "facebook/bart-large-cnn"


@lru_cache(maxsize=1)
def get_summarizer():
    """
    Load the summarization model once and reuse it.
    """
    return pipeline(
        task="summarization",
        model=MODEL_NAME,
        device=-1
    )


def summarize_text(text: str, max_sentences: int = 3) -> str:
    """
    Summarize text using BART, processing it in chunks.
    """
    if not text.strip():
        raise ValueError("Text cannot be empty.")

    chunks = chunk_text(
        text,
        max_words=250,
        overlap_words=30
    )

    summarizer = get_summarizer()
    summaries = []

    for chunk in chunks:
        result = summarizer(
            chunk,
            max_length=min(100, max(30, max_sentences * 20)),
            min_length=5,
            do_sample=False,
            truncation=True
        )

        summaries.append(result[0]["summary_text"])

    return " ".join(summaries)
