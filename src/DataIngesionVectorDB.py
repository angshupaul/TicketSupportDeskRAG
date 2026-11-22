"""
Step 1 of the Ticket Support Desk RAG pipeline:
Data Ingestion & Vector Store Setup.

- Loads support tickets from the JSON file in the Data folder
- Converts each ticket into a LangChain Document with metadata
- Embeds them with OpenAI `text-embedding-3-small`
- Stores them in a persistent Chroma vector store (reused on later runs)
"""

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "Data" / "synthetic_tickets_expanded.json"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "support_tickets"
EMBEDDING_MODEL = "text-embedding-3-small"
DISTANCE_METRIC = "cosine"  # HNSW space used to rank results (cosine similarity)


# --------------------------------------------------------------------------
# 1. Load tickets
# --------------------------------------------------------------------------
def load_tickets(file_path: Path = DATA_FILE) -> list[dict[str, Any]]:
    """Load the list of tickets from the JSON file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Tickets file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        tickets = json.load(f)

    print(f"Loaded {len(tickets)} tickets from {file_path.name}")
    return tickets


# --------------------------------------------------------------------------
# 2. Convert to LangChain Documents
# --------------------------------------------------------------------------
def ticket_to_document(ticket: dict[str, Any]) -> Document:
    """Build one Document per ticket.

    page_content -> title + description + resolution (the text that is embedded)
    metadata     -> fields used for filtering and citing sources
    """
    page_content = (
        f"Title: {ticket['title']}\n"
        f"Description: {ticket['description']}\n"
        f"Resolution: {ticket['resolution']}"
    )

    metadata = {
        "ticket_id": ticket["ticket_id"],
        "title": ticket["title"],
        "category": ticket["category"],
        "priority": ticket["priority"],
        "source": f"Ticket {ticket['ticket_id']}",
    }

    return Document(page_content=page_content, metadata=metadata)


def create_documents(tickets: list[dict[str, Any]]) -> list[Document]:
    """Convert all tickets into LangChain Documents."""
    documents = [ticket_to_document(t) for t in tickets]
    print(f"Created {len(documents)} documents")
    return documents


# --------------------------------------------------------------------------
# 3. Initialize OpenAI embeddings
# --------------------------------------------------------------------------
def get_embeddings() -> OpenAIEmbeddings:
    """Initialize OpenAI embeddings (requires OPENAI_API_KEY in env / .env)."""
    load_dotenv(PROJECT_ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Add it to a .env file in the project root."
        )
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


# --------------------------------------------------------------------------
# 4. Create / load Chroma vector store
# --------------------------------------------------------------------------
def get_distance_metric(vector_store: Chroma) -> str:
    """Return the HNSW distance metric ('cosine', 'l2' or 'ip') of the collection."""
    config = vector_store._collection.configuration_json or {}
    space = (config.get("hnsw") or {}).get("space")
    if space is None:  # older collections store it in metadata
        space = (vector_store._collection.metadata or {}).get("hnsw:space", "l2")
    return space


def build_vector_store(
    documents: list[Document] | None = None,
    embeddings: OpenAIEmbeddings | None = None,
) -> Chroma:
    """Open the persistent Chroma collection; embed and add documents only if it is empty.

    The collection uses cosine distance. If an existing collection was built with a
    different metric, it is reset and rebuilt so ranking is by cosine similarity.
    """
    embeddings = embeddings or get_embeddings()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_configuration={"hnsw": {"space": DISTANCE_METRIC}},
    )

    current_space = get_distance_metric(vector_store)
    if current_space != DISTANCE_METRIC:
        print(f"Collection uses '{current_space}' distance; rebuilding with '{DISTANCE_METRIC}'")
        vector_store.reset_collection()

    existing = vector_store._collection.count()
    if existing > 0:
        print(f"Loaded existing Chroma collection '{COLLECTION_NAME}' ({existing} vectors)")
        return vector_store

    if documents is None:
        documents = create_documents(load_tickets())

    # ticket_id as the vector id keeps entries unique and avoids duplicates
    ids = [doc.metadata["ticket_id"] for doc in documents]
    vector_store.add_documents(documents=documents, ids=ids)
    print(f"Created Chroma collection '{COLLECTION_NAME}' with {len(ids)} vectors at {CHROMA_DIR}")
    return vector_store


# --------------------------------------------------------------------------
# Run the ingestion pipeline
# --------------------------------------------------------------------------
def main() -> Chroma:
    tickets = load_tickets()
    documents = create_documents(tickets)

    print("\nSample document:")
    print(documents[0].page_content)
    print(documents[0].metadata)
    print()

    vector_store = build_vector_store(documents)

    # Quick sanity check
    query = "users cannot log in after password reset"
    results = vector_store.similarity_search(query, k=3)
    print(f"\nTest query: '{query}'")
    for doc in results:
        print(f"  - {doc.metadata['source']} | {doc.metadata['category']} | {doc.metadata['title']}")

    return vector_store


if __name__ == "__main__":
    main()
