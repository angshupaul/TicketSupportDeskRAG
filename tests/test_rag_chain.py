"""
Validation tests for the LCEL RAG chain (src/RAGChain.py).

Run from the project root:
    .venv\\Scripts\\python -m pytest tests/test_rag_chain.py -v

Note: most tests call OpenAI (embeddings + gpt-4o-mini), roughly a cent or less
per run. Answers are cached per question so each question is asked only once.
"""

import re
import sys
from functools import lru_cache
from pathlib import Path

import pytest
from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from RAGChain import (  # noqa: E402
    MIN_RELEVANCE_SCORE,
    build_rag_chain,
    build_rag_chain_with_sources,
    filter_by_score,
)
from SupportDeskAI import NO_ANSWER_MESSAGE  # noqa: E402

TICKET_ID_PATTERN = re.compile(r"TICK-\d{3}")

PASSWORD_RESET_Q = "Users cannot log in after a password reset. How was this fixed before?"
CRASHLOOP_Q = "Pods keep restarting with CrashLoopBackOff after a deployment. What should I check?"
OFF_TOPIC_QUESTIONS = ["How do I bake sourdough bread?", "What is the capital of France?"]


@lru_cache(maxsize=None)
def chain_with_sources():
    return build_rag_chain_with_sources()


@lru_cache(maxsize=None)
def ask_cached(question: str) -> dict:
    return chain_with_sources().invoke(question)


def allowed_ticket_ids(sources: list[dict]) -> set[str]:
    """Ticket IDs the model was shown: retrieved tickets plus their merged repeats."""
    ids = set()
    for s in sources:
        ids.add(s["ticket_id"])
        ids.update(filter(None, s["duplicate_ticket_ids"].split(",")))
    return ids


# --------------------------------------------------------------------------
# Scenario 1: the relevance gate drops low-scoring tickets (no API call)
# --------------------------------------------------------------------------
def test_relevance_gate_filters_by_score():
    docs = [
        Document(page_content="a", metadata={"ticket_id": "TICK-A", "score": 0.70}),
        Document(page_content="b", metadata={"ticket_id": "TICK-B", "score": MIN_RELEVANCE_SCORE}),
        Document(page_content="c", metadata={"ticket_id": "TICK-C", "score": 0.39}),
        Document(page_content="d", metadata={"ticket_id": "TICK-D"}),  # no score -> dropped
    ]
    kept = [d.metadata["ticket_id"] for d in filter_by_score(docs)]
    assert kept == ["TICK-A", "TICK-B"]
    assert filter_by_score(docs, min_score=0.8) == []


# --------------------------------------------------------------------------
# Scenario 2: the basic chain returns plain answer text with a citation
# --------------------------------------------------------------------------
def test_basic_chain_returns_cited_text():
    answer = build_rag_chain().invoke(
        "Database connection pool exhausted during peak hours. What fixed it?"
    )
    assert isinstance(answer, str)
    assert "TICK-002" in answer


# --------------------------------------------------------------------------
# Scenario 3: an answerable question cites the right ticket
# --------------------------------------------------------------------------
def test_answerable_question_cites_expected_ticket():
    result = ask_cached(PASSWORD_RESET_Q)
    assert "TICK-001" in result["answer"]
    assert NO_ANSWER_MESSAGE not in result["answer"]
    assert result["sources"][0]["ticket_id"] == "TICK-001"


# --------------------------------------------------------------------------
# Scenario 4: off-topic questions get the no-information answer, no sources
# --------------------------------------------------------------------------
@pytest.mark.parametrize("question", OFF_TOPIC_QUESTIONS)
def test_off_topic_question_returns_no_information(question):
    result = ask_cached(question)
    assert result["sources"] == []
    assert NO_ANSWER_MESSAGE in result["answer"]
    assert not TICKET_ID_PATTERN.search(result["answer"])


# --------------------------------------------------------------------------
# Scenario 5: hallucination check - every cited ticket was in the context
# --------------------------------------------------------------------------
@pytest.mark.parametrize("question", [PASSWORD_RESET_Q, CRASHLOOP_Q, "authentication issues"])
def test_citations_only_reference_retrieved_tickets(question):
    result = ask_cached(question)
    cited = set(TICKET_ID_PATTERN.findall(result["answer"]))
    assert cited, "answer should cite at least one ticket"
    assert cited <= allowed_ticket_ids(result["sources"])


# --------------------------------------------------------------------------
# Scenario 6: sources are complete, above the gate and sorted best first
# --------------------------------------------------------------------------
def test_sources_have_fields_and_are_sorted():
    result = ask_cached(CRASHLOOP_Q)
    sources = result["sources"]
    assert result["question"] == CRASHLOOP_Q
    assert 0 < len(sources) <= 5
    for s in sources:
        assert {"ticket_id", "title", "category", "priority", "score", "duplicate_ticket_ids"} <= s.keys()
        assert s["score"] >= MIN_RELEVANCE_SCORE
    scores = [s["score"] for s in sources]
    assert scores == sorted(scores, reverse=True)
    assert sources[0]["ticket_id"] == "TICK-037"


# --------------------------------------------------------------------------
# Scenario 7: an empty question is rejected before any LLM call
# --------------------------------------------------------------------------
@pytest.mark.parametrize("bad_question", ["", "   "])
def test_empty_question_raises_value_error(bad_question):
    with pytest.raises(ValueError):
        chain_with_sources().invoke(bad_question)
