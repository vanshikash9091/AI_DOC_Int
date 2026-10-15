
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

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
    word_count = len(request.text.split())

    return {
        "message": "Text received successfully",
        "word_count": word_count,
        "requested_summary_sentences": request.max_sentences,
        "summary": "AI summarization will be added in a later step."
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    allowed_types = {
        "text/plain",
        "application/pdf"
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail="Only TXT and PDF files are supported."
        )

    content = await file.read()

    if len(content) == 0:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty."
        )

    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="File size must not exceed 10 MB."
        )

    return {
        "message": "File uploaded successfully",
        "filename": file.filename,
        "content_type": file.content_type,
        "size_bytes": len(content)
    }
