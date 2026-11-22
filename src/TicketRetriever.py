"""
Step 2 of the Ticket Support Desk RAG pipeline: Retriever.

- Loads the persistent Chroma vector store built in DataIngesionVectorDB.py
- Ranks tickets by cosine similarity (the collection uses HNSW space='cosine')
- Removes repeated-title tickets (keeps the best-scoring one per title)
- Returns the top-5 most similar tickets, optionally filtered by metadata
"""

from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from DataIngesionVectorDB import DISTANCE_METRIC, build_vector_store, get_distance_metric

TOP_K = 5
FETCH_MULTIPLIER = 4  # candidates fetched = k * FETCH_MULTIPLIER, before removing repeats

_vector_store = None


def get_vector_store():
    """Load (once) the cosine-based Chroma vector store."""
    global _vector_store
    if _vector_store is None:
        _vector_store = build_vector_store()
        metric = get_distance_metric(_vector_store)
        if metric != DISTANCE_METRIC:
            raise RuntimeError(f"Vector store uses '{metric}' distance, expected '{DISTANCE_METRIC}'")
    return _vector_store


def dedupe_by_title(
    results: list[tuple[Document, float]], k: int
) -> list[tuple[Document, float]]:
    """Keep the best-scoring ticket per title (results must be sorted best first).

    The ids of dropped repeats are stored in the kept ticket's metadata as
    'duplicate_ticket_ids' (comma-separated string, empty if none).
    """
    kept: dict[str, tuple[Document, float]] = {}
    for doc, score in results:
        title = doc.metadata["title"]
        if title in kept:
            kept_doc = kept[title][0]
            ids = kept_doc.metadata["duplicate_ticket_ids"]
            kept_doc.metadata["duplicate_ticket_ids"] = ",".join(
                filter(None, [ids, doc.metadata["ticket_id"]])
            )
        else:
            doc.metadata["duplicate_ticket_ids"] = ""
            kept[title] = (doc, score)
    return list(kept.values())[:k]


def retrieve_with_scores(
    query: str,
    k: int = TOP_K,
    filter: dict[str, Any] | None = None,
    dedupe: bool = True,
) -> list[tuple[Document, float]]:
    """Return the top-k (Document, cosine_similarity) pairs, most similar first.

    Chroma returns cosine distance (0 = identical); cosine similarity = 1 - distance.
    With dedupe=True, tickets that repeat an already-returned title are skipped.
    """
    if not query or not query.strip():
        raise ValueError("Query must be a non-empty string")

    fetch_k = k * FETCH_MULTIPLIER if dedupe else k
    results = get_vector_store().similarity_search_with_score(query, k=fetch_k, filter=filter)
    scored = [(doc, 1.0 - distance) for doc, distance in results]
    return dedupe_by_title(scored, k) if dedupe else scored


class TicketRetriever(BaseRetriever):
    """LangChain retriever: top-k tickets by cosine similarity, repeats removed.

    The cosine similarity is added to each Document's metadata as 'score'.
    """

    k: int = TOP_K
    filter: dict[str, Any] | None = None
    dedupe: bool = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        results = retrieve_with_scores(query, k=self.k, filter=self.filter, dedupe=self.dedupe)
        for doc, score in results:
            doc.metadata["score"] = score
        return [doc for doc, _ in results]


def get_retriever(
    k: int = TOP_K, filter: dict[str, Any] | None = None, dedupe: bool = True
) -> TicketRetriever:
    """Return a LangChain retriever that gives the top-k tickets by cosine similarity.

    filter: optional Chroma metadata filter, e.g. {"category": "Database"}
            or {"$and": [{"category": "Database"}, {"priority": "Critical"}]}
    """
    get_vector_store()  # fail early if the store is missing or misconfigured
    return TicketRetriever(k=k, filter=filter, dedupe=dedupe)


def print_results(query: str, results: list[tuple[Document, float]]) -> None:
    print(f"\nQuery: '{query}'")
    for rank, (doc, score) in enumerate(results, start=1):
        m = doc.metadata
        dupes = f" | also: {m['duplicate_ticket_ids']}" if m.get("duplicate_ticket_ids") else ""
        print(f"  {rank}. {m['source']} | {m['category']} | {m['priority']} | "
              f"cosine={score:.4f} | {m['title']}{dupes}")


if __name__ == "__main__":
    retriever = get_retriever()
    docs = retriever.invoke("Database connection pool exhausted during peak hours")
    print(f"Retriever (top-{TOP_K}):", [d.metadata["ticket_id"] for d in docs])

    for q in [
        "authentication issues",
        #"users cannot log in after password reset",
        #"Pods keep restarting with CrashLoopBackOff after deployment",
        "customers charged twice for one order",
    ]:
        print_results(q, retrieve_with_scores(q))

    q = "certificate expired"
    print_results(q + "  [filter: category=SSL/TLS]",
                  retrieve_with_scores(q, filter={"category": "SSL/TLS"}))
