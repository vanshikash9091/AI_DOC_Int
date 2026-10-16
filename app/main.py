from app.text_processor import normalize_text, chunk_text
from io import BytesIO
from pathlib import Path
from app.summarizer import summarize_text

from fastapi import FastAPI, File, HTTPException, UploadFile, Form
from pydantic import BaseModel, Field
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.document_reader import extract_document_text

from app.qa_engine import answer_question
from app.upload_utils import read_limited_file

from app.database import save_document, list_documents, get_document

app = FastAPI(
    title="AI Document Intelligence API",
    description="An API for document analysis and AI-powered assistance.",
    version="1.0.0"
)


class SummarizeRequest(BaseModel):
    text: str = Field(min_length=20, max_length=10000)
    max_sentences: int = Field(default=3, ge=1, le=10)


@app.get("/")
def home():
    return {
        "message": "Welcome to AI Document Intelligence!",
        "status": "running"
    }


@app.post("/summarize")
def summarize_document(request: SummarizeRequest):
    try:
        summary = summarize_text(
            request.text,
            max_sentences=request.max_sentences
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {
        "message": "Summarization completed successfully",
        "word_count": len(request.text.split()),
        "requested_summary_sentences": request.max_sentences,
        "summary": summary
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    allowed_extensions = {".txt", ".pdf"}

    filename = file.filename or ""
    extension = Path(filename).suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=415,
            detail="Only TXT and PDF files are supported."
        )

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty."
        )

    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="File size must not exceed 10 MiB."
        )

    if extension == ".txt":
        try:
            extracted_text = content.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=400,
                detail="The TXT file must use UTF-8 encoding."
            )

        page_count = None

    else:
        try:
            reader = PdfReader(BytesIO(content))

            if reader.is_encrypted:
                raise HTTPException(
                    status_code=400,
                    detail="Password-protected PDFs are not supported."
                )

            page_count = len(reader.pages)

            extracted_text = "\n".join(
                page.extract_text() or ""
                for page in reader.pages
            )

        except PdfReadError:
            raise HTTPException(
                status_code=400,
                detail="The uploaded PDF could not be read."
            )

    if not extracted_text.strip():
        raise HTTPException(
            status_code=422,
            detail="No extractable text was found in the document."
        )
    # Clean the extracted text
    cleaned_text = normalize_text(extracted_text)

    # Divide the document into manageable chunks
    chunks = chunk_text(
        cleaned_text,
        max_words=400,
        overlap_words=50
    )

    return {
        "message": "Document processed successfully",
        "filename": Path(filename).name,
        "file_type": extension[1:].upper(),
        "page_count": page_count,
        "character_count": len(cleaned_text),
        "word_count": len(cleaned_text.split()),
        "chunk_count": len(chunks),
        "text_preview": cleaned_text[:1000],
        "first_chunk_preview": chunks[0][:500] if chunks else ""
    }

@app.post("/documents/summarize")
async def summarize_uploaded_document(
    file: UploadFile = File(...)
):
    filename = file.filename or ""

    try:
        content = await read_limited_file(file)
    finally:
        await file.close()

    document = extract_document_text(filename, content)

    cleaned_text = normalize_text(document["text"])

    if len(cleaned_text.split()) > 10000:
        raise HTTPException(
            status_code=413,
            detail="This version supports up to 10,000 words per document."
        )

    try:
        summary = summarize_text(cleaned_text, max_sentences=3)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc)
        )
    document_id = save_document(
    filename=Path(filename).name,
    file_type=document["file_type"],
    page_count=document["page_count"],
    word_count=len(cleaned_text.split()),
    summary=summary,
    )

    return {
        "message": "Document summarized successfully",
        "document_id": document_id,
        "filename": Path(filename).name,
        "file_type": document["file_type"],
        "page_count": document["page_count"],
        "word_count": len(cleaned_text.split()),
        "summary": summary
    }

@app.post("/documents/ask")
async def ask_document_question(
    file: UploadFile = File(...),
    question: str = Form(...)
):
    filename = file.filename or ""
    try:
        content = await read_limited_file(file)
    finally:
        await file.close()

    document = extract_document_text(filename, content)
    cleaned_text = normalize_text(document["text"])

    if len(cleaned_text.split()) > 10000:
        raise HTTPException(
            status_code=413,
            detail="This version supports up to 10,000 words per document."
        )

    try:
        result = answer_question(
            document_text=cleaned_text,
            question=question
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc)
        )

    return {
        "message": "Question processed successfully",
        "filename": Path(filename).name,
        "question": question,
        "answer": result["answer"],
        "confidence": result["confidence"],
        "source_passage": result["source_passage"]
    }
@app.get("/documents")
def get_documents():
    documents = list_documents()

    return {
        "count": len(documents),
        "documents": documents,
    }

@app.get("/documents/{document_id}")
def get_document_by_id(document_id: int):
    document = get_document(document_id)

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    return document
    
