
from app import qa_engine


def test_retrieves_matching_passage(monkeypatch):
    chunks = [
        "The company's revenue increased during 2025.",
        "The company hired several new employees.",
    ]

    monkeypatch.setattr(
        qa_engine,
        "chunk_text",
        lambda *args, **kwargs: chunks,
    )

    results = qa_engine.retrieve_chunks(
        "ignored test document",
        "What was the revenue in 2025?",
        top_k=3,
    )

    assert results == [chunks[0]]


def test_excludes_irrelevant_passages(monkeypatch):
    chunks = [
        "The company hired several new employees.",
        "The office moved to a larger building.",
    ]

    monkeypatch.setattr(
        qa_engine,
        "chunk_text",
        lambda *args, **kwargs: chunks,
    )

    results = qa_engine.retrieve_chunks(
        "ignored test document",
        "What was the revenue?",
    )

    assert results == []


def test_rare_query_term_can_rank_higher(monkeypatch):
    chunks = [
        "The company expanded its operations.",
        "The acquisition was completed successfully.",
        "The company hired new employees.",
    ]

    monkeypatch.setattr(
        qa_engine,
        "chunk_text",
        lambda *args, **kwargs: chunks,
    )

    results = qa_engine.retrieve_chunks(
        "ignored test document",
        "company acquisition",
        top_k=3,
    )

    assert results[0] == chunks[1]


def test_empty_document_returns_no_passages(monkeypatch):
    monkeypatch.setattr(
        qa_engine,
        "chunk_text",
        lambda *args, **kwargs: [],
    )

    results = qa_engine.retrieve_chunks(
        "",
        "revenue",
    )

    assert results == []


def test_unsearchable_question_returns_no_passages(monkeypatch):
    chunks = [
        "The company revenue increased.",
        "The business hired employees.",
    ]

    monkeypatch.setattr(
        qa_engine,
        "chunk_text",
        lambda *args, **kwargs: chunks,
    )

    results = qa_engine.retrieve_chunks(
        "ignored test document",
        "what is the",
    )

    assert results == []


def test_top_k_limits_number_of_results(monkeypatch):
    chunks = [
        "Revenue increased during the year.",
        "Revenue increased during the quarter.",
        "Revenue increased during the month.",
    ]

    monkeypatch.setattr(
        qa_engine,
        "chunk_text",
        lambda *args, **kwargs: chunks,
    )

    results = qa_engine.retrieve_chunks(
        "ignored test document",
        "revenue increased",
        top_k=2,
    )

    assert len(results) == 2
    assert all("revenue" in chunk.lower() for chunk in results)
