# Data Ingestion & Vector Store Notes

Step 1 of the Ticket Support Desk RAG pipeline.
Code: [src/DataIngesionVectorDB.py](src/DataIngesionVectorDB.py)

## Source data
- File: `Data/synthetic_tickets_expanded.json`
- 294 tickets, all with the same 8 fields: `ticket_id`, `title`, `description`, `resolution`, `category`, `priority`, `created_date`, `resolved_date`
- 36 categories (e.g. Database 21, Security 20, Kubernetes 18, Authentication 16)
- Priorities: High 141, Medium 76, Critical 65, Low 12
- Tickets are short: description + resolution is at most about 550 characters

## Design decisions
| Decision | Choice |
|---|---|
| Embedded text (`page_content`) | Title + Description + Resolution |
| Metadata | `ticket_id`, `title`, `category`, `priority`, `source` (`"Ticket <ticket_id>"`) |
| Chunking | None: one Document per ticket (tickets are short) |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector store | Chroma, saved to disk at `./chroma_db`, collection `support_tickets` |
| Rebuild behaviour | Reuse the collection if it already has vectors; embed only when it is empty |
| Vector ids | `ticket_id`, so a ticket can't be stored twice |
| API key | `OPENAI_API_KEY` loaded from `.env` in the project root |

## Pipeline functions
1. **`load_tickets()`**: reads the JSON file and returns the list of tickets.
2. **`ticket_to_document()` / `create_documents()`**: turns each ticket into a LangChain `Document`:
   - `page_content` = `Title: ...\nDescription: ...\nResolution: ...`
   - `metadata` = the fields listed above
3. **`get_embeddings()`**: loads `.env`, checks that `OPENAI_API_KEY` is set, and returns `OpenAIEmbeddings(model="text-embedding-3-small")`.
4. **`build_vector_store()`**: opens the Chroma collection. If it has vectors, returns it as is; otherwise embeds and adds all Documents.
5. **`main()`**: prints a sample Document, builds or loads the store, and runs a test search (`"users cannot log in after password reset"`, k=3).

## Setup & run
- Virtual environment: `.venv` with **Python 3.12** (3.14 is the system default, but some packages like faiss-cpu, chromadb and llama-index may not have 3.14 wheels yet)
- `OPENAI_API_KEY` is set in `.env` in the project root

```bash
py -3.12 -m venv .venv                 # one-time
.venv\Scripts\activate                 # PowerShell / cmd
pip install -r requirements.txt
python src/DataIngesionVectorDB.py
```

Key installed versions: langchain 1.5.2, langchain-core 1.8.4, langchain-openai 1.8.5, langchain-chroma 1.1.0, chromadb 1.5.9, langgraph 1.2.24, llama-index 0.14.26, faiss-cpu 1.16.0, openai 3.57.0.

## Things to remember
- `langchain-chroma` was added to requirements.txt because the code imports `langchain_chroma`.
- Loading the tickets and building the Documents was tested (294 Documents). The embedding and Chroma step hasn't been run yet.
- **Delete `./chroma_db`** after changing the `page_content` or metadata format; otherwise the old collection is reused.
- Later steps can `from DataIngesionVectorDB import build_vector_store` to get the existing store without re-embedding.
- Metadata supports filtered search, e.g. `vector_store.similarity_search(query, k=3, filter={"category": "Database"})`.

## Vector store check (2026-11-21)
- The `./chroma_db` collection `support_tickets` holds 294 vectors of 1536 dimensions (about 4.4 MB on disk).
- Test query `"users cannot log in after password reset"` returns TICK-001 first, which is correct.
- **Indexing:** Chroma builds an HNSW index automatically (`space=l2`, `ef_construction=100`, `ef_search=100`, `max_neighbors=16`). No extra indexing is needed:
  - With 294 documents, HNSW search is essentially exact and fast.
  - OpenAI embeddings are unit-length, so L2 and cosine give the same ranking.
  - Metadata filters (`category`, `priority`) work without a separate index.
- Optional: create the collection with `collection_metadata={"hnsw:space": "cosine"}` so raw scores read as cosine distance. This only changes how scores read, not the ranking. It needs `./chroma_db` deleted and rebuilt.
- **Repeated titles:** 134 titles are shared by 268 tickets (e.g. TICK-194 and TICK-284, "Password reset links expired on arrival"). Their descriptions and resolutions differ, so they are separate incidents, not duplicates. Top-k results may still contain near-identical tickets, so consider MMR retrieval in step 2.

## Step 2: Retriever (cosine similarity, top-5, repeats removed)
Code: [src/TicketRetriever.py](src/TicketRetriever.py) · Tests: [tests/test_retriever.py](tests/test_retriever.py)

- **Cosine switch:** the collection is created with `collection_configuration={"hnsw": {"space": "cosine"}}`. `build_vector_store()` checks the metric of the existing collection; if it isn't cosine, it resets and re-embeds the collection automatically. The old L2 store was rebuilt this way on 2026-11-21.
- **`TOP_K = 5`** (changed from 3 on 2026-11-22).
- **Removing repeated titles:** 134 titles are shared by two tickets each, so plain top-k returned near-identical pairs.
  - `retrieve_with_scores()` fetches `k * FETCH_MULTIPLIER` (5 × 4 = 20) candidates.
  - It keeps the best-scoring ticket per title and returns the top 5.
  - The IDs of dropped repeats go in the kept ticket's metadata as `duplicate_ticket_ids` (comma-separated, e.g. `"TICK-105"`). That's useful for "this issue happened N times" in the RAG answer.
  - Pass `dedupe=False` to turn this off.
  - A filtered search can return fewer than 5 results. For example, SSL/TLS has only 4 distinct titles.
- **`get_retriever(k=5, filter=None, dedupe=True)`** returns a `TicketRetriever` (a custom LangChain `BaseRetriever`) that can be used directly in a RAG chain. Each Document also gets `metadata["score"]` (its cosine similarity).
- **`retrieve_with_scores(query, k=5, filter=None, dedupe=True)`** returns `(Document, cosine_similarity)` pairs, most similar first. Chroma returns cosine *distance*, so similarity = `1 - distance`. An empty query raises `ValueError`.
- Filter examples: `{"category": "SSL/TLS"}`, `{"$and": [{"category": "Database"}, {"priority": "Critical"}]}`
- Typical scores: a strong match is about 0.65–0.75 and an unrelated filler result is about 0.38–0.52. Vague queries such as "authentication issues" only reach about 0.42–0.47, so plan a minimum-score / "no match" fallback for the RAG step.

### Tests (pytest, 16 cases across 10 scenarios, all passing)
Run: `.venv\Scripts\python -m pytest tests -v` (calls the OpenAI embeddings API for the queries, a fraction of a cent per run)
1. Collection uses cosine and holds 294 tickets
2. Retriever returns exactly `TOP_K` (5) distinct Documents, each with a `score`
3. Known issue (password reset login) returns TICK-001 first, and the top 3 are all Authentication
4. Queries map to the right category: Kubernetes, Payment, Database, Email, SSL/TLS
5. Scores are sorted descending and match a cosine similarity recomputed with numpy from the stored vectors
6. Metadata filters: a single category, and category + priority combined (up to `TOP_K` results)
7. Documents carry the expected metadata keys and the Title/Resolution content
8. A ticket's exact title finds that ticket (TICK-003, similarity > 0.7)
9. Empty and whitespace-only queries raise `ValueError`
10. Repeated titles are removed: the CrashLoopBackOff query keeps TICK-037 (the best score) and records TICK-105 in `duplicate_ticket_ids`

### Project hygiene
- `.gitignore` covers `.env`, `.venv/`, `chroma_db/`, `__pycache__/`, `.pytest_cache/` and editor files. Create it before any `git init`, so the API key is never committed.

## Next step
Step 3: RAG chain (LLM answer generation from the retrieved tickets).
