# Ticket Support Desk RAG

A Retrieval-Augmented Generation (RAG) assistant that helps engineers troubleshoot issues using historical support tickets. Ask a question in plain English and **SupportDesk AI** answers using only past tickets, citing the ticket IDs it used. If no ticket is relevant, it says so instead of guessing.

```
You: Users cannot log in after a password reset. How was this fixed before?

Assistant: The issue was addressed in ticket [TICK-001]. The password hash algorithm
was updated but session tokens weren't invalidated. The fix was to clear all active
sessions and force re-authentication...

📎 Sources: TICK-001, TICK-284, TICK-082, ...
```

## How it works

```
question → retriever → relevance gate → prompt → LLM → answer + sources
```

1. **Ingestion:** 294 support tickets (`Data/synthetic_tickets_expanded.json`) become LangChain Documents. Each one's title, description and resolution is embedded with OpenAI `text-embedding-3-small` and stored in a persistent **Chroma** vector store.
2. **Retrieval:** finds the top 5 tickets by **cosine similarity** and removes tickets that repeat the same title.
3. **Relevance gate:** drops tickets scoring below 0.40, so off-topic questions reach the model with no context.
4. **Prompt:** anti-hallucination rules tell the model to answer only from the provided tickets, cite every ticket ID, and say *"I don't have enough information…"* when the tickets don't cover the question.
5. **Generation:** `gpt-4o-mini` at temperature 0, assembled into a chain with **LCEL** (LangChain Expression Language).

## Project structure

| Path | Purpose |
|---|---|
| `src/DataIngesionVectorDB.py` | Load tickets, create Documents, build or load the Chroma vector store |
| `src/TicketRetriever.py` | Top-5 cosine-similarity retriever with repeated titles removed |
| `src/SupportDeskAI.py` | Prompt template with anti-hallucination rules, context formatting, LLM setup |
| `src/RAGChain.py` | LCEL RAG chain, relevance gate, answer + sources |
| `src/SupportDeskChat.py` | Interactive terminal chat |
| `tests/` | pytest suites for the retriever and the RAG chain |
| `*notes.md` | Beginner-level notes for each step (see below) |

## Setup

Requires **Python 3.12** (some dependencies may not have wheels for 3.14 yet) and an OpenAI API key.

```bash
git clone https://github.com/angshupaul/TicketSupportDeskRAG.git
cd TicketSupportDeskRAG

py -3.12 -m venv .venv
.venv\Scripts\activate          # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt

copy .env.example .env          # macOS/Linux: cp .env.example .env
# then edit .env and set OPENAI_API_KEY
```

## Usage

```bash
# 1. Build the vector store (one time; re-embedding 294 tickets costs a fraction of a cent)
python src/DataIngesionVectorDB.py

# 2. Chat with SupportDesk AI
python src/SupportDeskChat.py
```

Type a question, or `quit` to exit.

To use it from Python:

```python
from RAGChain import ask            # run from src/, or add src/ to sys.path

result = ask("Pods keep restarting with CrashLoopBackOff. What should I check?")
print(result["answer"])             # cited answer
print(result["sources"])            # tickets used: id, title, category, priority, score
```

## Tests

```bash
python -m pytest tests -v
```

27 tests cover:
- **Retrieval:** cosine ranking, top-k, category mapping, metadata filters, removal of repeated titles.
- **The RAG chain:**
  - correct citations;
  - off-topic questions refused;
  - every cited ticket ID was actually retrieved (a hallucination check);
  - sources returned with each answer.

The tests call the OpenAI API, roughly a cent or less per run.

## Notes

- [dataandvectornotes.md](dataandvectornotes.md): data ingestion, vector store and retriever
- [SupportDeskNotes.md](SupportDeskNotes.md): prompt template and LLM
- [RAGChainnotes.md](RAGChainnotes.md): the LCEL RAG chain, explained step by step

## Roadmap

- Evaluation: retrieval hit rate and mean reciprocal rank, answer quality with ROUGE or an LLM grader
- Streaming answers in the chat
- LangGraph agent: category routing, clarifying questions for vague queries
