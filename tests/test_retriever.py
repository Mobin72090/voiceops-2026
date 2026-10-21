
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retriever import load_documents, search_documents


def test_sections_keep_source_metadata():
    chunks = load_documents()

    assert chunks, "No knowledge-base sections were loaded."

    for chunk in chunks:
        assert chunk["source"]
        assert chunk["section_id"]
        assert chunk["status"]
        assert chunk["effective_from"]


def test_electronics_returns_retrieves_current_policy():
    results = search_documents(
        "What is the return window for electronics?",
        top_k=5,
    )

    section_ids = {item["section_id"] for item in results}

    assert "RET-2" in section_ids or "RET-3" in section_ids
    assert all(item["status"].lower() != "superseded" for item in results)


def test_return_window_counting_retrieves_calendar_days():
    results = search_documents(
        "How are return windows counted?",
        top_k=5,
    )

    section_ids = {item["section_id"] for item in results}

    assert "RET-2" in section_ids or "KB-00.3" in section_ids
