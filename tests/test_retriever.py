"""
Validation tests for the ticket retriever (src/TicketRetriever.py).

Run from the project root:
    .venv\\Scripts\\python -m pytest tests -v

Note: these tests call the OpenAI embeddings API to embed the queries
(a fraction of a cent per run). The vector store must already be built.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from TicketRetriever import TOP_K, get_retriever, get_vector_store, retrieve_with_scores  # noqa: E402

EXPECTED_METADATA_KEYS = {"ticket_id", "title", "category", "priority", "source"}


# --------------------------------------------------------------------------
# Scenario 1: the vector store is set up for cosine similarity
# --------------------------------------------------------------------------
def test_vector_store_uses_cosine_and_has_all_tickets():
    vs = get_vector_store()
    config = vs._collection.configuration_json
    assert config["hnsw"]["space"] == "cosine"
    assert vs._collection.count() == 294


# --------------------------------------------------------------------------
# Scenario 2: the retriever returns exactly the top-k Documents
# --------------------------------------------------------------------------
def test_retriever_returns_top_k_documents():
    docs = get_retriever().invoke("Database connection pool exhausted during peak hours")
    assert TOP_K == 5
    assert len(docs) == TOP_K
    assert len({d.metadata["ticket_id"] for d in docs}) == TOP_K  # no repeated tickets
    assert all("score" in d.metadata for d in docs)


# --------------------------------------------------------------------------
# Scenario 3: a known problem returns the matching ticket first
# --------------------------------------------------------------------------
def test_known_issue_returns_matching_ticket_first():
    results = retrieve_with_scores("users cannot log in after password reset")
    top_doc, top_score = results[0]
    assert top_doc.metadata["ticket_id"] == "TICK-001"
    # with 5 distinct results, the lower ranks may come from nearby categories
    assert all(doc.metadata["category"] == "Authentication" for doc, _ in results[:3])
    assert top_score > 0.6


# --------------------------------------------------------------------------
# Scenario 4: queries from different areas map to the right category
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "query, expected_category",
    [
        ("Pods keep restarting with CrashLoopBackOff after deployment", "Kubernetes"),
        ("customers were charged twice for a single purchase", "Payment"),
        ("database connection timeout, connection pool exhausted", "Database"),
        ("our emails are going to the Gmail spam folder", "Email"),
        ("TLS certificate expired and browsers show a security warning", "SSL/TLS"),
    ],
)
def test_query_top_result_has_expected_category(query, expected_category):
    top_doc, _ = retrieve_with_scores(query)[0]
    assert top_doc.metadata["category"] == expected_category


# --------------------------------------------------------------------------
# Scenario 5: results are ranked by cosine similarity, highest first,
# and the scores match a cosine similarity computed independently
# --------------------------------------------------------------------------
def test_results_ranked_by_cosine_similarity():
    query = "API requests are being rate limited with 429 errors"
    results = retrieve_with_scores(query)
    scores = [score for _, score in results]

    assert scores == sorted(scores, reverse=True)
    assert all(0.0 < s <= 1.0 for s in scores)

    # Recompute cosine similarity with numpy from the stored vectors
    vs = get_vector_store()
    query_vec = np.array(vs._embedding_function.embed_query(query))
    ids = [doc.metadata["ticket_id"] for doc, _ in results]
    stored = vs._collection.get(ids=ids, include=["embeddings"])
    vec_by_id = dict(zip(stored["ids"], stored["embeddings"]))

    for doc, score in results:
        doc_vec = np.array(vec_by_id[doc.metadata["ticket_id"]])
        cosine = query_vec @ doc_vec / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
        assert cosine == pytest.approx(score, abs=1e-3)


# --------------------------------------------------------------------------
# Scenario 6: metadata filters restrict the results
# --------------------------------------------------------------------------
def test_category_filter_restricts_results():
    docs = get_retriever(filter={"category": "SSL/TLS"}).invoke("certificate expired")
    # SSL/TLS has only 4 distinct titles, so dedupe can return fewer than TOP_K
    assert 0 < len(docs) <= TOP_K
    assert all(d.metadata["category"] == "SSL/TLS" for d in docs)


def test_combined_category_and_priority_filter():
    flt = {"$and": [{"category": "Database"}, {"priority": "Critical"}]}
    results = retrieve_with_scores("database is slow and running out of space", filter=flt)
    assert 0 < len(results) <= TOP_K
    for doc, _ in results:
        assert doc.metadata["category"] == "Database"
        assert doc.metadata["priority"] == "Critical"


# --------------------------------------------------------------------------
# Scenario 7: returned Documents carry content and metadata for citing
# --------------------------------------------------------------------------
def test_documents_have_content_and_metadata():
    for doc, _ in retrieve_with_scores("mobile app crashes on startup"):
        assert EXPECTED_METADATA_KEYS <= doc.metadata.keys()
        assert doc.metadata["source"] == f"Ticket {doc.metadata['ticket_id']}"
        assert doc.page_content.startswith("Title: ")
        assert "Resolution: " in doc.page_content


# --------------------------------------------------------------------------
# Scenario 8: a ticket's own title finds that ticket with high similarity
# --------------------------------------------------------------------------
def test_exact_title_query_finds_ticket():
    results = retrieve_with_scores("Payment processing fails for international cards")
    top_doc, top_score = results[0]
    assert top_doc.metadata["ticket_id"] == "TICK-003"
    assert top_score > 0.7


# --------------------------------------------------------------------------
# Scenario 9: invalid input is rejected
# --------------------------------------------------------------------------
@pytest.mark.parametrize("bad_query", ["", "   "])
def test_empty_query_raises_value_error(bad_query):
    with pytest.raises(ValueError):
        retrieve_with_scores(bad_query)


# --------------------------------------------------------------------------
# Scenario 10: repeated-title tickets are removed, keeping the best score,
# and the dropped ticket ids are recorded on the kept ticket
# --------------------------------------------------------------------------
def test_dedupe_removes_repeated_titles():
    query = "Pods keep restarting with CrashLoopBackOff after deployment"

    raw = retrieve_with_scores(query, dedupe=False)
    raw_titles = [doc.metadata["title"] for doc, _ in raw]
    assert len(raw_titles) != len(set(raw_titles))  # repeats exist without dedupe

    results = retrieve_with_scores(query)
    titles = [doc.metadata["title"] for doc, _ in results]
    assert len(results) == TOP_K
    assert len(titles) == len(set(titles))  # no repeats with dedupe

    top_doc, top_score = results[0]
    assert top_doc.metadata["ticket_id"] == "TICK-037"
    assert "TICK-105" in top_doc.metadata["duplicate_ticket_ids"].split(",")
    assert top_score == pytest.approx(raw[0][1], abs=1e-3)  # best-scoring copy is kept
