import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from catalog import (
    HIGH_VALUE_THRESHOLD,
    get_product_policy_expansion,
    load_products,
    lookup_product,
)
from retriever import (
    compute_kb_hash,
    cosine_similarity,
    get_embedded_chunks,
    load_cached_embeddings,
    load_documents,
    save_cached_embeddings,
    search_documents,
    search_documents_lexical,
    search_documents_semantic,
)


def test_sections_keep_source_metadata():
    """Verify KB loading preserves all required metadata fields."""
    chunks = load_documents()

    assert chunks, "No knowledge-base sections were loaded."

    for chunk in chunks:
        assert chunk["source"]
        assert chunk["section_id"]
        assert chunk["status"]
        assert chunk["effective_from"]
        assert chunk["doc_type"]
        assert chunk["text"]


def test_electronics_returns_retrieves_current_policy():
    """Verify standard queries retrieve current return policy without superseded docs."""
    results = search_documents(
        "What is the return window for electronics?",
        top_k=5,
    )

    section_ids = {item["section_id"] for item in results}

    assert "RET-2" in section_ids or "RET-3" in section_ids
    assert all(item["status"].lower() != "superseded" for item in results)


def test_return_window_counting_retrieves_calendar_days():
    """Verify return window counting logic references calendar days."""
    results = search_documents(
        "How are return windows counted?",
        top_k=5,
    )

    section_ids = {item["section_id"] for item in results}

    assert "RET-2" in section_ids or "KB-00.3" in section_ids


def test_semantic_ranking_prefers_semantically_relevant_chunk():
    """Verify semantic ranking ranks higher cosine similarity chunks at the top."""
    chunk_a = {
        "section_id": "SEM-A",
        "heading": "Alpha Topic",
        "source": "alpha.md",
        "doc_type": "policy",
        "status": "current",
        "effective_from": "2026-08-01",
        "text": "Alpha content.",
        "embedding": [1.0, 0.0],
    }
    chunk_b = {
        "section_id": "SEM-B",
        "heading": "Beta Topic",
        "source": "beta.md",
        "doc_type": "policy",
        "status": "current",
        "effective_from": "2026-08-01",
        "text": "Beta content.",
        "embedding": [0.0, 1.0],
    }

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_emb = MagicMock()
    mock_emb.values = [1.0, 0.0]
    mock_resp.embeddings = [mock_emb]
    mock_client.models.embed_content.return_value = mock_resp

    results = search_documents_semantic(
        question="Alpha question",
        top_k=2,
        chunks=[chunk_b, chunk_a],
        client=mock_client,
    )

    assert results is not None
    assert len(results) == 2
    assert results[0]["section_id"] == "SEM-A"
    assert results[0]["score"] > results[1]["score"]
    assert "embedding" not in results[0]


def test_lexical_fallback_when_api_key_missing():
    """Verify transparent fallback to lexical retrieval when API key is missing."""
    with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
        with patch("retriever.get_embedding_client", return_value=None):
            results = search_documents(
                "What is the return window for electronics?",
                top_k=5,
                mode="hybrid",
                client=None,
            )

            assert results, "Should return results via lexical fallback."
            section_ids = {item["section_id"] for item in results}
            assert "RET-2" in section_ids or "RET-3" in section_ids
            assert all(item["status"].lower() != "superseded" for item in results)


def test_lexical_fallback_when_embedding_call_fails():
    """Verify transparent fallback to lexical retrieval if the embedding API throws an error."""
    mock_client = MagicMock()
    mock_client.models.embed_content.side_effect = RuntimeError("Rate limit 429 quota exhausted")

    results = search_documents(
        "What is the return window for electronics?",
        top_k=5,
        mode="hybrid",
        client=mock_client,
    )

    assert results, "Should fallback to lexical results when API fails."
    section_ids = {item["section_id"] for item in results}
    assert "RET-2" in section_ids or "RET-3" in section_ids


def test_cache_validity_and_reuse(tmp_path):
    """Verify that cached embeddings are saved and loaded without re-embedding."""
    test_cache = tmp_path / "test_cache.json"

    sample_chunks = [
        {
            "section_id": "TEST-1",
            "heading": "Test Section",
            "source": "test.md",
            "doc_type": "policy",
            "status": "current",
            "effective_from": "2026-08-01",
            "text": "Some test text.",
            "embedding": [0.1, 0.2, 0.3],
        }
    ]

    save_cached_embeddings(
        sample_chunks,
        cache_path=test_cache,
        model_name="gemini-embedding-001",
    )

    loaded = load_cached_embeddings(
        cache_path=test_cache,
        expected_model="gemini-embedding-001",
    )

    assert loaded is not None
    assert len(loaded) == 1
    assert loaded[0]["section_id"] == "TEST-1"
    assert loaded[0]["embedding"] == [0.1, 0.2, 0.3]


def test_cache_invalidation_when_kb_hash_mismatched(tmp_path):
    """Verify cache is rejected when the stored hash differs from the current KB hash."""
    test_cache = tmp_path / "stale_cache.json"

    stale_payload = {
        "version": 1,
        "model": "gemini-embedding-001",
        "kb_hash": "outdated_or_stale_hash_value_9999",
        "chunks": [
            {
                "section_id": "STALE-1",
                "heading": "Stale",
                "embedding": [0.1, 0.2],
            }
        ],
    }

    import json
    with open(test_cache, "w", encoding="utf-8") as f:
        json.dump(stale_payload, f)

    loaded = load_cached_embeddings(
        cache_path=test_cache,
        expected_model="gemini-embedding-001",
    )

    assert loaded is None, "Stale cache must be rejected and invalidated."


def test_historical_policy_filtering():
    """Verify superseded policies are excluded for standard queries but retrieved for historical queries."""
    std_results = search_documents_lexical(
        "What is the return window for electronics?",
        top_k=10,
    )
    for r in std_results:
        assert r["status"].lower() != "superseded"

    hist_results = search_documents_lexical(
        "What was the return window under previous policy version 3 before 1 august 2026?",
        top_k=10,
    )
    has_superseded = any(r["status"].lower() == "superseded" for r in hist_results)
    assert has_superseded, "Historical query should be able to retrieve superseded policy sections."


def test_cosine_similarity_edge_cases():
    """Verify cosine similarity correctly handles orthogonal, identical, and zero vectors."""
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0
    assert cosine_similarity([], []) == 0.0


# --- Product Catalog & Cross-Referencing Unit Tests ---

def test_catalog_explicit_sku_lookup():
    """Verify explicit SKU detection and verified product fact retrieval."""
    product = lookup_product("The Kestrel Book 14 Laptop (SKU ELE-0023) has a crack.")
    assert product is not None
    assert product["sku"] == "ELE-0023"
    assert product["price_inr"] == 45999
    assert product["return_category"] == "electronics"
    assert product["price_inr"] > HIGH_VALUE_THRESHOLD


def test_catalog_unambiguous_product_title_matching():
    """Verify reliable title lookup when SKU is omitted."""
    product = lookup_product("My Kestrel Book 14 Laptop has a cracked screen.")
    assert product is not None
    assert product["brand"] == "Kestrel"
    assert product["return_category"] == "electronics"
    assert product["price_inr"] > HIGH_VALUE_THRESHOLD


def test_catalog_ambiguous_product_name_not_guessed():
    """Verify that product names whose variants cross the ₹10,000 threshold are not guessed."""
    # In products.csv, "Aangan Engineered Wood Study Table" has Blue at ₹10,639 and Grey at ₹9,239.
    # Without SKU or color, the high-value classification is ambiguous and must return None.
    product = lookup_product("My Aangan Engineered Wood Study Table is broken.")
    assert product is None, "Ambiguous product name straddling price threshold must not be guessed."


def test_catalog_unknown_sku_not_guessed():
    """Verify unknown SKUs return None and do not trigger hallucinated product facts."""
    product = lookup_product("I received item SKU XYZ-9999 with defects.")
    assert product is None


def test_catalog_missing_file_handling(tmp_path):
    """Verify missing catalog CSV is handled gracefully without crashing."""
    fake_path = tmp_path / "non_existent_products.csv"
    empty_cat = load_products(fake_path)
    assert empty_cat == {}
    assert lookup_product("SKU ELE-0023", catalog_path=fake_path) is None


def test_catalog_exact_10000_boundary():
    """Verify exact ₹10,000 policy boundary from DWM-3 & ESC-1 (threshold: price_inr > 10000)."""
    mock_catalog = {
        "TST-EXACT": {
            "sku": "TST-EXACT",
            "title": "Exact Ten Thousand Item",
            "price_inr": 10000,
            "return_category": "electronics",
        },
        "TST-ABOVE": {
            "sku": "TST-ABOVE",
            "title": "Above Ten Thousand Item",
            "price_inr": 10001,
            "return_category": "electronics",
        },
        "TST-BELOW": {
            "sku": "TST-BELOW",
            "title": "Below Ten Thousand Item",
            "price_inr": 9999,
            "return_category": "electronics",
        },
    }

    # Exactly ₹10,000: price_inr > 10000 is False (standard damage, no DWM-3/ESC-1)
    exp_exact = get_product_policy_expansion("Item SKU TST-EXACT is cracked", catalog=mock_catalog)
    assert "DWM-1" in exp_exact and "DWM-2" in exp_exact
    assert "DWM-3" not in exp_exact and "ESC-1" not in exp_exact

    # ₹10,001: price_inr > 10000 is True (high-value damage escalation DWM-3 & ESC-1)
    exp_above = get_product_policy_expansion("Item SKU TST-ABOVE is cracked", catalog=mock_catalog)
    assert "DWM-3" in exp_above and "ESC-1" in exp_above

    # ₹9,999: price_inr > 10000 is False (standard damage)
    exp_below = get_product_policy_expansion("Item SKU TST-BELOW is cracked", catalog=mock_catalog)
    assert "DWM-1" in exp_below and "DWM-2" in exp_below
    assert "DWM-3" not in exp_below and "ESC-1" not in exp_below


def test_retrieval_high_value_damaged_product():
    """Verify high-value damaged product query retrieves DWM-3 and ESC-1."""
    results = search_documents(
        "The Kestrel Book 14 Laptop (SKU ELE-0023) I received yesterday has a crack. I want it sorted.",
        top_k=5,
    )
    section_ids = {r["section_id"] for r in results}
    assert "DWM-3" in section_ids or "ESC-1" in section_ids


def test_retrieval_lower_value_damaged_product():
    """Verify lower-value damaged product query retrieves DWM-1/DWM-2 and avoids DWM-3."""
    results = search_documents(
        "Is the collection free if the UrbanThread Slim Fit Jeans came torn?",
        top_k=5,
    )
    section_ids = {r["section_id"] for r in results}
    assert "DWM-1" in section_ids or "DWM-2" in section_ids
    assert "DWM-3" not in section_ids, "Lower-value item must not trigger high-value escalation."


def test_retrieval_unknown_sku_fallback():
    """Verify queries with unknown SKUs fall back gracefully to standard retrieval."""
    results = search_documents("My item with SKU XYZ-9999 arrived damaged.", top_k=5)
    assert len(results) > 0


def test_retrieval_non_product_query():
    """Verify standard non-product queries are unaffected by catalog cross-referencing."""
    results = search_documents("How are return windows counted?", top_k=5)
    section_ids = {r["section_id"] for r in results}
    assert "RET-2" in section_ids or "KB-00.3" in section_ids
