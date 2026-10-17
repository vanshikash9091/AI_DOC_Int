
import sqlite3
from pathlib import Path

# Store the database in the main project folder.
DB_PATH = Path(__file__).resolve().parent.parent / "documents.db"


def get_connection():
    """Create and return a database connection."""
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection



def init_db():
    """Create the documents table and apply simple schema upgrades."""
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                file_type TEXT NOT NULL,
                page_count INTEGER,
                word_count INTEGER NOT NULL,
                summary TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(documents)"
            ).fetchall()
        }

        if "extracted_text" not in columns:
            connection.execute(
                """
                ALTER TABLE documents
                ADD COLUMN extracted_text TEXT NOT NULL DEFAULT ''
                """
            )



def save_document(
    filename: str,
    file_type: str,
    page_count: int | None,
    word_count: int,
    summary: str,
    extracted_text: str,
) -> int:
    """Save document metadata, summary, and extracted text."""
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO documents
                (
                    filename,
                    file_type,
                    page_count,
                    word_count,
                    summary,
                    extracted_text
                )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                filename,
                file_type,
                page_count,
                word_count,
                summary,
                extracted_text,
            ),
        )
        return cursor.lastrowid


def list_documents() -> list[dict]:
    """Return all saved document records, newest first."""
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, filename, file_type, page_count,
                   word_count, summary, created_at
            FROM documents
            ORDER BY id DESC
            """
        ).fetchall()

        return [dict(row) for row in rows]


def get_document(document_id: int) -> dict | None:
    """Return one document record, or None if it does not exist."""
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT id, filename, file_type, page_count,
                   word_count, summary, created_at
            FROM documents
            WHERE id = ?
            """,
            (document_id,),
        ).fetchone()

        return dict(row) if row is not None else None

def get_document_text(document_id: int) -> dict | None:
    """Retrieve the stored text and metadata for one document."""
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT id, filename, file_type, word_count, extracted_text
            FROM documents
            WHERE id = ?
            """,
            (document_id,),
        ).fetchone()

        return dict(row) if row is not None else None

# Ensure the table exists when this module is imported.
init_db()
