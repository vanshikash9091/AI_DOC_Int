
import pytest

from app import database


@pytest.fixture(autouse=True)
def isolate_test_database(tmp_path, monkeypatch):
    # Give each test its own temporary SQLite database.
    test_db_path = tmp_path / "test_documents.db"

    # Redirect database operations to the temporary database.
    monkeypatch.setattr(database, "DB_PATH", test_db_path)

    # Create the required database table and columns.
    database.init_db()
