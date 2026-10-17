
from fastapi.testclient import TestClient

import app.main as main

client = TestClient(main.app)


def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_summarize_endpoint(monkeypatch):
    def fake_summarize_text(text, max_sentences=3):
        return "This is a test summary."

    monkeypatch.setattr(
        main,
        "summarize_text",
        fake_summarize_text
    )

    response = client.post(
        "/summarize",
        json={
            "text": (
                "FastAPI makes it easy to build APIs. "
                "It supports validation and documentation."
            ),
            "max_sentences": 3
        }
    )

    assert response.status_code == 200
    assert response.json()["summary"] == "This is a test summary."


def test_upload_txt():
    content = (
        b"FastAPI builds APIs. Document processing extracts text."
    )

    response = client.post(
        "/upload",
        files={
            "file": ("sample.txt", content, "text/plain")
        }
    )

    assert response.status_code == 200
    assert response.json()["filename"] == "sample.txt"


def test_upload_rejects_unsupported_file():
    response = client.post(
        "/upload",
        files={
            "file": (
                "sample.exe",
                b"example content",
                "application/octet-stream"
            )
        }
    )

    assert response.status_code == 415


def test_document_summarization(monkeypatch):
    def fake_summarize_text(text, max_sentences=3):
        return "A mocked document summary."

    monkeypatch.setattr(
        main,
        "summarize_text",
        fake_summarize_text
    )

    content = (
        b"FastAPI is a Python framework. "
        b"It supports document processing and APIs."
    )

    response = client.post(
        "/documents/summarize",
        files={
            "file": ("report.txt", content, "text/plain")
        }
    )

    assert response.status_code == 200
    assert response.json()["summary"] == "A mocked document summary."


def test_document_question_answering(monkeypatch):
    def fake_answer_question(document_text, question):
        return {
            "answer": "FastAPI builds APIs.",
            "confidence": 0.95,
            "source_passage": document_text
        }

    monkeypatch.setattr(
        main,
        "answer_question",
        fake_answer_question
    )

    content = b"FastAPI builds APIs and supports document processing."

    response = client.post(
        "/documents/ask",
        data={"question": "What does FastAPI build?"},
        files={
            "file": ("report.txt", content, "text/plain")
        }
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "FastAPI builds APIs."
    assert response.json()["source_passage"] is not None


def test_document_is_saved_and_can_be_retrieved(monkeypatch):
    def fake_summarize_text(text, max_sentences=3):
        return "A saved test summary."

    monkeypatch.setattr(main, "summarize_text", fake_summarize_text)

    content = (
        b"This is a test document. "
        b"It contains enough text for summarization."
    )

    response = client.post(
        "/documents/summarize",
        files={"file": ("saved_test.txt", content, "text/plain")},
    )

    assert response.status_code == 200
    document_id = response.json()["document_id"]

    detail_response = client.get(f"/documents/{document_id}")

    assert detail_response.status_code == 200
    assert detail_response.json()["filename"] == "saved_test.txt"
    assert detail_response.json()["summary"] == "A saved test summary."


def test_document_list_contains_saved_document(monkeypatch):
    def fake_summarize_text(text, max_sentences=3):
        return "A list test summary."

    monkeypatch.setattr(main, "summarize_text", fake_summarize_text)

    content = b"This document checks the document listing endpoint."

    create_response = client.post(
        "/documents/summarize",
        files={"file": ("list_test.txt", content, "text/plain")},
    )

    assert create_response.status_code == 200

    response = client.get("/documents")

    assert response.status_code == 200
    assert response.json()["count"] >= 1

    filenames = [
        document["filename"]
        for document in response.json()["documents"]
    ]
    assert "list_test.txt" in filenames


def test_missing_document_returns_404():
    response = client.get("/documents/999999")

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found."


def test_ask_saved_document(monkeypatch):
    def fake_summarize_text(text, max_sentences=3):
        return "A stored document summary."

    def fake_answer_question(document_text, question):
        assert "FastAPI builds APIs" in document_text
        assert question == "What does FastAPI build?"

        return {
            "answer": "APIs",
            "confidence": 0.95,
            "source_passage": "FastAPI builds APIs.",
        }

    monkeypatch.setattr(main, "summarize_text", fake_summarize_text)
    monkeypatch.setattr(main, "answer_question", fake_answer_question)

    content = b"FastAPI builds APIs and supports document processing."

    create_response = client.post(
        "/documents/summarize",
        files={"file": ("qa_saved.txt", content, "text/plain")},
    )

    assert create_response.status_code == 200
    document_id = create_response.json()["document_id"]

    response = client.post(
        f"/documents/{document_id}/ask",
        json={"question": "What does FastAPI build?"},
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "APIs"
    assert response.json()["document_id"] == document_id


def test_ask_saved_document_not_found():
    response = client.post(
        "/documents/999999/ask",
        json={"question": "What does this document say?"},
    )

    assert response.status_code == 404


def test_ask_saved_document_without_text(monkeypatch):
    from app.database import save_document

    document_id = save_document(
        filename="old_record.txt",
        file_type="TXT",
        page_count=None,
        word_count=5,
        summary="An older summary.",
        extracted_text="",
    )

    response = client.post(
        f"/documents/{document_id}/ask",
        json={"question": "What does this document say?"},
    )

    assert response.status_code == 409
