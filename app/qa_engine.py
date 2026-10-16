
import re
from functools import lru_cache

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
    chunks = chunk_text(
        document_text,
        max_words=180,
        overlap_words=30
    )

    question_words = tokenize(question)

    if not question_words:
        return chunks[:top_k]

    scored_chunks = []

    for chunk in chunks:
        chunk_words = tokenize(chunk)
        score = len(question_words & chunk_words)
        scored_chunks.append((score, chunk))

    scored_chunks.sort(key=lambda item: item[0], reverse=True)

    return [
        chunk
        for score, chunk in scored_chunks[:top_k]
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
