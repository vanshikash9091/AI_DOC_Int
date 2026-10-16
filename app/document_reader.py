
from io import BytesIO
from pathlib import Path

from fastapi import HTTPException
from pypdf import PdfReader
from pypdf.errors import PdfReadError


def extract_document_text(filename: str, content: bytes) -> dict:
    extension = Path(filename).suffix.lower()

    if extension not in {".txt", ".pdf"}:
        raise HTTPException(
            status_code=415,
            detail="Only TXT and PDF files are supported."
        )

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
            text = content.decode("utf-8")
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
            text = "\n".join(
                page.extract_text() or ""
                for page in reader.pages
            )

        except PdfReadError:
            raise HTTPException(
                status_code=400,
                detail="The uploaded PDF could not be read."
            )

    if not text.strip():
        raise HTTPException(
            status_code=422,
            detail="No extractable text was found in the document."
        )

    return {
        "text": text,
        "file_type": extension[1:].upper(),
        "page_count": page_count
    }
