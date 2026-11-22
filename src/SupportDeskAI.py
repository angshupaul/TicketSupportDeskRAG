"""
Step 3 of the Ticket Support Desk RAG pipeline: SupportDesk AI.

- Prompt template with anti-hallucination rules
- Formats retrieved tickets into the {context} block (with ticket IDs for citation)
- Initializes the OpenAI chat model (OPENAI_CHAT_MODEL, default 'gpt-4o-mini')
"""

import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from DataIngesionVectorDB import PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_CHAT_MODEL = "gpt-4o-mini"
NO_ANSWER_MESSAGE = (
    "I don't have enough information in the ticket history to answer that question."
)

# --------------------------------------------------------------------------
# 1. Prompt template with anti-hallucination rules
# --------------------------------------------------------------------------
prompt_template = """You are SupportDesk AI, a technical support assistant that helps engineers troubleshoot issues using historical support ticket data.

CRITICAL RULES:
1. Answer ONLY using information from the provided context
2. If the answer is not in the context, say "I don't have enough information in the ticket history to answer that question."
3. DO NOT make up information or use external knowledge
4. Always cite the ticket ID when referencing information
5. If multiple tickets are relevant, mention all of them

Context from support tickets:
{context}

Question: {question}
Helpful Answer (with ticket citations):"""

SUPPORT_DESK_PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"],
)


# --------------------------------------------------------------------------
# 2. Format retrieved tickets into the {context} block
# --------------------------------------------------------------------------
def format_context(docs: list[Document]) -> str:
    """Turn retrieved ticket Documents into a context block the LLM can cite.

    Each ticket is labelled with its ID, category and priority. Tickets that were
    merged as repeats of the same title (see TicketRetriever.dedupe_by_title) are
    listed so the model can mention them too.
    """
    if not docs:
        return "No relevant tickets found."

    blocks = []
    for doc in docs:
        m = doc.metadata
        header = f"[{m['ticket_id']}] Category: {m['category']} | Priority: {m['priority']}"
        if m.get("duplicate_ticket_ids"):
            header += f" | Same issue also seen in: {m['duplicate_ticket_ids']}"
        blocks.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


# --------------------------------------------------------------------------
# 3. Initialize the language model
# --------------------------------------------------------------------------
def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    """Initialize the OpenAI chat model.

    Model comes from OPENAI_CHAT_MODEL in .env, defaulting to 'gpt-4o-mini'.
    temperature=0 keeps answers deterministic and close to the context.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set. Add it to the .env file.")

    model_name = os.getenv("OPENAI_CHAT_MODEL", DEFAULT_CHAT_MODEL)
    return ChatOpenAI(model=model_name, temperature=temperature)


if __name__ == "__main__":
    from TicketRetriever import get_retriever

    llm = get_llm()
    print(f"LLM initialized: {llm.model_name} (temperature={llm.temperature})")

    question = "Users cannot log in after a password reset. How was this fixed before?"
    docs = get_retriever().invoke(question)
    prompt_text = SUPPORT_DESK_PROMPT.format(context=format_context(docs), question=question)

    print("\n===== Formatted prompt =====")
    print(prompt_text)
