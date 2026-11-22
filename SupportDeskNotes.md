# SupportDesk AI Notes (Step 3)

Beginner-level explanation of [src/SupportDeskAI.py](src/SupportDeskAI.py) and how it fits into the RAG system.
For steps 1–2 (ingestion, vector store, retriever), see [dataandvectornotes.md](dataandvectornotes.md).

## Where this file fits in the RAG system

RAG stands for **Retrieval-Augmented Generation**, which comes down to three jobs:

| Letter | Job | Where it happens |
|---|---|---|
| **R**etrieval | Find past tickets that match the question | Steps 1–2: `DataIngesionVectorDB.py`, `TicketRetriever.py` |
| **A**ugmented | Put those tickets into the AI's instructions | **`SupportDeskAI.py`** (prompt + `format_context`) |
| **G**eneration | Have the AI write the answer | **`SupportDeskAI.py`** (`get_llm`) |

The earlier steps built a search engine over old tickets. This file prepares everything needed to turn those search results into a helpful, honest answer.

**Analogy:** picture a new support engineer at your company.
- The **retriever** is a colleague who pulls the 5 most relevant old tickets from the filing cabinet.
- The **prompt** is the instruction sheet you give the new engineer: "Only use these tickets, don't guess, say which ticket you got each fact from."
- The **LLM** is the engineer, who reads the tickets and writes the reply.

---

## Step by step through the code

### Step 1: Load settings
```python
load_dotenv(PROJECT_ROOT / ".env")
```
This reads the `.env` file so the code can find the **OpenAI API key** (your account password for OpenAI) and, optionally, which model to use. Keeping these in `.env` keeps secrets out of the code.

### Step 2: The prompt template, the AI's instruction sheet
```python
prompt_template = """You are SupportDesk AI ...
CRITICAL RULES: ...
Context from support tickets:
{context}
Question: {question}
Helpful Answer (with ticket citations):"""
```
A **prompt** is the text sent to the AI. A **template** is a prompt with blanks to fill in:
- `{context}`: filled with the tickets the retriever found
- `{question}`: filled with what the user asked

**Why the "CRITICAL RULES"?** AI models sometimes **hallucinate**: they invent answers that sound confident but are wrong. In a support desk, a made-up fix could make things worse. Each rule fights that:

| Rule | What it prevents |
|---|---|
| 1. Use ONLY the context | The AI answering from general internet knowledge instead of *your* tickets |
| 2. Say "I don't have enough information…" | Guessing when no ticket covers the question |
| 3. Don't make things up | Invented fixes, settings or numbers |
| 4. Cite the ticket ID | Lets an engineer check the original ticket (e.g. `TICK-001`) |
| 5. Mention all relevant tickets | Missing a second, equally useful past fix |

```python
SUPPORT_DESK_PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
```
This wraps the text in a LangChain `PromptTemplate` object, so later you can call `.format(context=..., question=...)` and the blanks get filled in.

`NO_ANSWER_MESSAGE` stores the "not enough information" sentence on its own. Later code (or tests) can then check whether the AI declined to answer.

### Step 3: `format_context()` turns search results into readable text
The retriever returns Python **Document** objects, but the AI only reads plain text. This function converts each ticket into a labelled block:

```
[TICK-284] Category: Authentication | Priority: High | Same issue also seen in: TICK-194
Title: Password reset links expired on arrival
Description: ...
Resolution: ...
---
[TICK-001] ...
```
Why it's built this way:
- **`[TICK-284]` at the top** gives the AI the exact ID to cite (rule 4).
- **Category and priority** add useful background, e.g. that it was a high-priority auth issue.
- **"Same issue also seen in"**: the retriever merged tickets with identical titles, and this line tells the AI the problem happened more than once (rule 5).
- **`---` between tickets** keeps the AI from mixing up details from different tickets.
- **No tickets found** returns `"No relevant tickets found."`, so the AI sees nothing to use and follows rule 2.

### Step 4: `get_llm()` sets up the AI model
```python
model_name = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
return ChatOpenAI(model=model_name, temperature=0.0)
```
- **LLM** (Large Language Model) is the AI that writes the answer. Here it's OpenAI's `gpt-4o-mini`, which is fast and cheap.
- `os.getenv(..., "gpt-4o-mini")` means: use the model named in `.env` if there is one, otherwise `gpt-4o-mini`. You can switch models without touching the code. (`OPENAI_CHAT_MODEL` is currently commented out in `.env`, so `gpt-4o-mini` is used.)
- **`temperature=0`** controls creativity. 0 means the AI gives the most likely, consistent answer every time, which is what you want for factual support answers. Higher values make it more creative, and more likely to make things up.
- It first checks that the API key exists and gives a clear error if it doesn't.

### Step 5: The demo at the bottom (`if __name__ == "__main__":`)
This only runs when you execute the file directly (`python src/SupportDeskAI.py`). It:
1. Creates the LLM and prints which model it is
2. Asks the retriever for tickets about a password-reset question
3. Fills the template with those tickets and the question
4. Prints the **final prompt**, exactly what the AI would receive

It doesn't ask the AI for an answer yet. It just shows what the AI would be given.

---

## The whole flow, start to finish

```
User question
     │
     ▼
Retriever (Step 2) ──► top 5 similar tickets (Documents)
     │
     ▼
format_context() ──► labelled plain-text ticket blocks        ← this file
     │
     ▼
SUPPORT_DESK_PROMPT ──► rules + tickets + question             ← this file
     │
     ▼
get_llm() → gpt-4o-mini ──► answer with ticket citations       ← this file (set up)
```

## Status (2026-11-22)
- The prompt formats correctly with retrieved tickets. A test call to the LLM returned "OK" from `gpt-4o-mini-2024-07-18`.
- **What's missing:** the arrows aren't connected yet. Each piece exists, but nothing yet takes a question, runs it through all of them, and returns the answer. That's the next step: building the **RAG chain**.
- **Something to watch:** for the password-reset question, the 4th and 5th tickets are only loosely related (account lockouts, botnet brute-force). Rule 1 should stop the AI from treating them as answers, but a minimum similarity score (around 0.40) before building the context would make that more reliable.
