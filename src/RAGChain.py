"""
Step 4 of the Ticket Support Desk RAG pipeline: RAG chain built with LCEL.

Assembles the pieces from the earlier steps with the | (pipe) operator:

    question -> retriever -> relevance gate -> format_context -> prompt -> LLM -> text

- build_rag_chain()              : returns the answer text only
- build_rag_chain_with_sources() : returns {"question", "answer", "sources"}
- ask(question)                  : one-line helper using the chain with sources
"""

from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel, RunnablePassthrough

from SupportDeskAI import SUPPORT_DESK_PROMPT, format_context, get_llm
from TicketRetriever import TOP_K, get_retriever

MIN_RELEVANCE_SCORE = 0.40  # tickets below this cosine similarity are dropped


# --------------------------------------------------------------------------
# Relevance gate: drop tickets that are not similar enough to the question
# --------------------------------------------------------------------------
def filter_by_score(docs: list[Document], min_score: float = MIN_RELEVANCE_SCORE) -> list[Document]:
    """Keep only tickets whose cosine similarity (metadata['score']) is >= min_score.

    If nothing passes, an empty list is returned, so format_context() produces
    "No relevant tickets found." and the LLM answers with the no-information message.
    """
    return [doc for doc in docs if doc.metadata.get("score", 0.0) >= min_score]


def docs_to_sources(docs: list[Document]) -> list[dict[str, Any]]:
    """Summarise the tickets used as context, best first, for display or citation checks."""
    return [
        {
            "ticket_id": d.metadata["ticket_id"],
            "title": d.metadata["title"],
            "category": d.metadata["category"],
            "priority": d.metadata["priority"],
            "score": round(d.metadata["score"], 4),
            "duplicate_ticket_ids": d.metadata.get("duplicate_ticket_ids", ""),
        }
        for d in docs
    ]


# --------------------------------------------------------------------------
# Chain 1: basic RAG chain (question in, answer text out)
# --------------------------------------------------------------------------
def build_rag_chain(k: int = TOP_K, min_score: float = MIN_RELEVANCE_SCORE) -> Runnable:
    retriever = get_retriever(k=k)
    relevance_gate = RunnableLambda(lambda docs: filter_by_score(docs, min_score))

    return (
        {
            # path 1: find tickets -> drop weak ones -> turn them into text
            "context": retriever | relevance_gate | format_context,
            # path 2: pass the question through unchanged
            "question": RunnablePassthrough(),
        }
        | SUPPORT_DESK_PROMPT   # fill in {context} and {question}
        | get_llm()             # the AI writes the answer
        | StrOutputParser()     # keep just the text
    )


# --------------------------------------------------------------------------
# Chain 2: RAG chain that also returns the source tickets
# --------------------------------------------------------------------------
def build_rag_chain_with_sources(k: int = TOP_K, min_score: float = MIN_RELEVANCE_SCORE) -> Runnable:
    retriever = get_retriever(k=k)
    relevance_gate = RunnableLambda(lambda docs: filter_by_score(docs, min_score))

    # Step A: retrieve once and keep both the documents and the question
    retrieve = RunnableParallel(
        docs=retriever | relevance_gate,
        question=RunnablePassthrough(),
    )

    # Step B: generate the answer from those documents
    generate_answer = (
        RunnableLambda(lambda x: {"context": format_context(x["docs"]), "question": x["question"]})
        | SUPPORT_DESK_PROMPT
        | get_llm()
        | StrOutputParser()
    )

    # Step C: add the answer, then shape the final output
    return (
        retrieve
        | RunnablePassthrough.assign(answer=generate_answer)
        | RunnableLambda(lambda x: {
            "question": x["question"],
            "answer": x["answer"],
            "sources": docs_to_sources(x["docs"]),
        })
    )


_default_chain: Runnable | None = None


def ask(question: str) -> dict[str, Any]:
    """Answer a question with the default chain (built once): {"question", "answer", "sources"}."""
    global _default_chain
    if _default_chain is None:
        _default_chain = build_rag_chain_with_sources()
    return _default_chain.invoke(question)


def print_response(result: dict[str, Any]) -> None:
    print(f"\nQ: {result['question']}")
    print(f"A: {result['answer']}")
    if result["sources"]:
        print("Sources:")
        for s in result["sources"]:
            dupes = f" (also: {s['duplicate_ticket_ids']})" if s["duplicate_ticket_ids"] else ""
            print(f"  - {s['ticket_id']} | score={s['score']:.2f} | {s['title']}{dupes}")
    else:
        print("Sources: none passed the relevance gate")


if __name__ == "__main__":
    # Basic chain: just the answer text
    basic = build_rag_chain()
    print("Basic chain answer:")
    print(basic.invoke("Database connection pool exhausted during peak hours. What fixed it?"))

    # Chain with sources
    for q in [
        "Users cannot log in after a password reset. How was this fixed before?",
        "Pods keep restarting with CrashLoopBackOff after a deployment. What should I check?",
        "authentication issues",
        "How do I bake sourdough bread?",
    ]:
        print_response(ask(q))
