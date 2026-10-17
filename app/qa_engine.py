
import re
from functools import lru_cache
import math
from transformers import pipeline

from app.text_processor import chunk_text


QA_MODEL_NAME = "distilbert/distilbert-base-cased-distilled-squad"

STOP_WORDS = {
    "what", "when", "where", "who", "why", "how",
    "is", "are", "was", "were", "the", "a", "an",
    "of", "to", "in", "on", "for", "and", "or",
    "does", "do", "did", "can", "could", "would"
}


def tokenize(text: str) -> set[str]:
    return {
        word.lower()
        for word in re.findall(r"\b\w+\b", text)
        if word.lower() not in STOP_WORDS
    }



def retrieve_chunks(
    document_text: str,
    question: str,
    top_k: int = 3
) -> list[str]:
    # Split the document into overlapping passages.
    chunks = chunk_text(
        document_text,
        max_words=180,
        overlap_words=30
    )

    # Do not retrieve passages if the limit is invalid.
    if top_k <= 0 or not chunks:
        return []

    # Convert the question into searchable words.
    question_words = tokenize(question)

    # No searchable words means we cannot rank meaningfully.
    if not question_words:
        return []

    # Count how many chunks contain each word.
    document_frequency = {}

    for chunk in chunks:
        chunk_words = tokenize(chunk)

        for word in chunk_words:
            document_frequency[word] = (
                document_frequency.get(word, 0) + 1
            )

    # Assign more weight to words appearing in fewer chunks.
    total_chunks = len(chunks)
    word_weights = {
        word: math.log(
            (total_chunks + 1) /
            (document_frequency.get(word, 0) + 1)
        ) + 1
        for word in question_words
    }

    # Calculate a normalized relevance score for each chunk.
    total_question_weight = sum(word_weights.values())
    scored_chunks = []

    for position, chunk in enumerate(chunks):
        chunk_words = tokenize(chunk)

        matched_weight = sum(
            word_weights[word]
            for word in question_words
            if word in chunk_words
        )

        score = matched_weight / total_question_weight

        scored_chunks.append((score, position, chunk))

    # Rank by score, preserving original order when scores tie.
    scored_chunks.sort(
        key=lambda item: (-item[0], item[1])
    )

    # Return only relevant passages, up to the requested limit.
    return [
        chunk
        for score, position, chunk in scored_chunks[:top_k]
        if score > 0
    ]


@lru_cache(maxsize=1)
def get_qa_pipeline():
    return pipeline(
        task="question-answering",
        model=QA_MODEL_NAME,
        device=-1
    )


def answer_question(
    document_text: str,
    question: str
) -> dict:
    if not document_text.strip():
        raise ValueError("Document text cannot be empty.")

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    chunks = retrieve_chunks(document_text, question)

    if not chunks:
        return {
            "answer": "I could not find a relevant passage in this document.",
            "confidence": 0.0,
            "source_passage": None
        }

    qa_model = get_qa_pipeline()
    candidates = []

    for chunk in chunks:
        result = qa_model(
            question=question,
            context=chunk
        )

        candidates.append({
            "answer": result["answer"],
            "confidence": float(result["score"]),
            "source_passage": chunk
        })

    best = max(candidates, key=lambda item: item["confidence"])

    if best["confidence"] < 0.15 or not best["answer"].strip():
        return {
            "answer": "I could not identify a reliable answer in this document.",
            "confidence": best["confidence"],
            "source_passage": best["source_passage"]
        }

    return best
