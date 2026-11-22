# RAG Chain Notes (Step 4, LCEL)

Beginner-level notes for [src/RAGChain.py](src/RAGChain.py) and its tests [tests/test_rag_chain.py](tests/test_rag_chain.py).
Earlier steps: [dataandvectornotes.md](dataandvectornotes.md) (ingestion, vector store, retriever) · [SupportDeskNotes.md](SupportDeskNotes.md) (prompt and LLM).

## What this step does
The earlier steps built separate pieces: a retriever, a context formatter, a prompt and an LLM. This step **connects them** so that one call takes a question and returns a cited answer:

```
question → retriever → relevance gate → format_context → prompt → LLM → answer text
```

## What is LCEL?
**LCEL (LangChain Expression Language)** connects pieces with the `|` (pipe) symbol, like an assembly line:

```python
chain = step_a | step_b | step_c
answer = chain.invoke("my question")
```
The output of `step_a` is handed to `step_b`, then to `step_c`. Each piece is a **Runnable**, meaning it has an `.invoke()` method that takes an input and returns an output. The retriever, prompt and LLM are already Runnables. A plain Python function becomes one by wrapping it in `RunnableLambda(...)`.

| Piece | From | What it does |
|---|---|---|
| `get_retriever()` | Step 2 | question → up to 5 ticket Documents (repeats removed, each with a `score`) |
| `filter_by_score()` | **this step** | drops tickets scoring below 0.40 (the "relevance gate") |
| `format_context()` | Step 3 | Documents → labelled plain text |
| `SUPPORT_DESK_PROMPT` | Step 3 | context + question → full prompt with anti-hallucination rules |
| `get_llm()` | Step 3 | prompt → AI message (`gpt-4o-mini`, temperature 0) |
| `StrOutputParser()` | **this step** | AI message → just the answer text |

---

## Step by step through the code

### Step 1: The relevance gate, `filter_by_score()`
The retriever **always** returns up to 5 tickets, even for "How do I bake bread?". The gate keeps only tickets whose cosine similarity is at least `MIN_RELEVANCE_SCORE = 0.40`.

| Kind of question | Typical scores | Result |
|---|---|---|
| Strong match (e.g. CrashLoopBackOff) | 0.65–0.75 | passes |
| Vague but related ("authentication issues") | 0.42–0.47 | passes |
| Off-topic ("bake bread", "capital of France", "set up a laptop") | 0.07–0.23 | dropped |

If nothing passes, `format_context([])` returns `"No relevant tickets found."` and the AI follows prompt rule 2, replying *"I don't have enough information in the ticket history to answer that question."*

This is the **second safeguard** against hallucination: the prompt rules are the first, and the gate keeps unrelated tickets from reaching the AI at all.

### Step 2: Chain 1, the basic chain, `build_rag_chain()`
```python
{
    "context": retriever | relevance_gate | format_context,   # path 1
    "question": RunnablePassthrough(),                         # path 2
}
| SUPPORT_DESK_PROMPT
| get_llm()
| StrOutputParser()
```
- **The `{...}` dictionary at the start.** The prompt needs **two** inputs (`context` and `question`), but the user gives **one** (the question). The dictionary sends the question down two paths at once:
  - path 1 finds tickets, drops weak ones and turns them into text;
  - path 2 (`RunnablePassthrough()`) passes the question along untouched.
  Both results meet at the prompt.
- **`StrOutputParser()`.** The AI returns a message object with extra data (token counts, etc.). This keeps just the text.
- **Returns:** a plain string, the answer.

### Step 3: Chain 2, with sources, `build_rag_chain_with_sources()`
Engineers want to check the original tickets, so this version returns the answer **and** the tickets it was based on. It has three parts:

- **A, `retrieve`** (`RunnableParallel`): runs the retriever and gate once and keeps `{"docs": [...], "question": "..."}`.
- **B, `generate_answer`**: turns `docs` into context text, then prompt → LLM → text.
- **C, `RunnablePassthrough.assign(answer=generate_answer)`**: "keep what I have and add an `answer` field". A final `RunnableLambda` shapes the output:

```python
{
  "question": "Users cannot log in after a password reset...",
  "answer":   "...addressed in ticket [TICK-001]. ...clearing all active sessions...",
  "sources":  [{"ticket_id": "TICK-001", "title": "...", "category": "Authentication",
                "priority": "High", "score": 0.65, "duplicate_ticket_ids": ""}, ...]
}
```
Why retrieve only once? If the answer and the sources were fetched separately, they would cost two searches and might not match. Retrieving once guarantees that the sources are exactly what the AI saw.

### Step 4: Easy entry point, `ask(question)`
It builds the chain with sources the first time it's called, then reuses it. One line gives you the full result:
```python
from RAGChain import ask
result = ask("Pods keep restarting with CrashLoopBackOff. What should I check?")
print(result["answer"])
```
`print_response(result)` prints the answer with its source list.

### Step 5: Demo, `python src/RAGChain.py`
This runs the basic chain once, then the chain with sources on 4 questions: password reset, CrashLoopBackOff, "authentication issues" and "bake sourdough bread" (off-topic).

---

## Results from the demo (2026-11-22)
| Question | Answer cited | Sources | Notes |
|---|---|---|---|
| DB connection pool exhausted | TICK-002 | n/a (basic chain) | correct fix: max_connections 20 → 100 |
| Password reset login | TICK-001 | 5 (top: TICK-001, 0.65) | correct fix: clear sessions, force re-authentication |
| CrashLoopBackOff | TICK-037 | 5 (top: TICK-037, 0.74) | lists liveness/startup probe checks from the ticket |
| "authentication issues" | TICK-106, 001, 016, 109 | 5 | sensibly **skipped** TICK-055 (a Payment ticket) |
| Bake sourdough bread | none | 0 (all dropped by the gate) | exact "I don't have enough information…" reply |

A spot check of the TICK-106 summary against the real ticket found it accurate, with nothing invented.

## Tests: 11 cases across 7 scenarios, all passing
Run: `.venv\Scripts\python -m pytest tests -v`. That's all 27 tests, including the 16 retriever tests.
Cost: calls OpenAI embeddings + `gpt-4o-mini`, roughly a cent or less per run. Answers are cached per question inside the test file, so each question is asked once.

1. **Relevance gate:** keeps scores ≥ 0.40, drops lower or missing scores. This runs without any API call.
2. **Basic chain:** returns a plain string citing TICK-002.
3. **Answerable question:** the password-reset question cites TICK-001, and TICK-001 is the top source.
4. **Off-topic questions** ("bake bread", "capital of France") give no sources, the exact no-information message, and no ticket IDs.
5. **Hallucination check:** every `TICK-xxx` cited in the answer must be a retrieved ticket or one of its merged repeats (tested on 3 questions).
6. **Sources:** all fields present, all scores ≥ 0.40, sorted best first, top is TICK-037 for CrashLoopBackOff.
7. **Empty question** is rejected with `ValueError` before any LLM call.

## Things to remember
- **Tuning the gate:** change `MIN_RELEVANCE_SCORE` in `RAGChain.py`, or pass `min_score=` to the build functions. A higher value refuses more often; a lower value lets in weaker tickets.
- **The AI's wording can vary slightly** between runs even at temperature 0. The tests check for ticket IDs and the exact refusal sentence, not full answer text.
- **A question can still pass the gate on topic words alone.** For example, "Kafka consumer group rebalancing" scores 0.63 because a related ticket exists. The prompt rules are what stop the AI from going beyond that ticket's content.

## Possible next steps
1. **Evaluation:** build 20–30 questions with their expected ticket IDs and measure retrieval quality (hit rate, mean reciprocal rank) plus answer quality (ROUGE against the real `resolution`, or an LLM grader). Use it to tune `TOP_K` and `MIN_RELEVANCE_SCORE` with numbers.
2. **Streaming:** use `chain.stream(question)` to print the answer word by word, for a chat-like feel.
3. **LangGraph agent:** detect the category or priority of a question and filter on it, or ask a clarifying question when the query is vague.
