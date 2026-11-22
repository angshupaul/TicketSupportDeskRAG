"""
Interactive SupportDesk chat: try the whole RAG pipeline from the terminal.

Run from the project root:
    .venv\\Scripts\\python src/SupportDeskChat.py
"""

import os
import sys

from dotenv import load_dotenv

from DataIngesionVectorDB import PROJECT_ROOT

# Make sure emoji / box characters print on any Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv(PROJECT_ROOT / ".env")

# ============================================================================
# Set up the RAG chain
# ============================================================================
qa_chain = None
if os.getenv("OPENAI_API_KEY"):
    from RAGChain import build_rag_chain_with_sources

    # Returns {"question", "answer", "sources"}: the sources are exactly the
    # tickets the AI was given (after the relevance gate), from a single search.
    qa_chain = build_rag_chain_with_sources()

# ============================================================================
# Interactive Demo
# ============================================================================
print("\n" + "="*80)
print("Interactive SupportDesk Assistant")
print("="*80)

if qa_chain:
    print("\nSupportDesk RAG Assistant Ready!")
    print("Ask questions about support ticket history.")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            user_query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):  # Ctrl+Z / Ctrl+C
            print("\nGoodbye!")
            break

        if user_query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if not user_query:
            continue

        print("\nAssistant: ", end="", flush=True)
        try:
            result = qa_chain.invoke(user_query)
        except Exception as e:  # e.g. network or API errors: keep the chat running
            print(f"⚠ Something went wrong: {e}\n")
            continue

        print(result["answer"])

        sources = result["sources"]
        if sources:
            print(f"\n📎 Sources: {', '.join(s['ticket_id'] for s in sources)}")
            for s in sources:
                dupes = f" (also: {s['duplicate_ticket_ids']})" if s["duplicate_ticket_ids"] else ""
                print(f"   - {s['ticket_id']} | score={s['score']:.2f} | {s['title']}{dupes}")
        else:
            print("\n📎 Sources: none (no ticket was similar enough to the question)")
        print()
else:
    print("\n⚠ Interactive mode requires OpenAI API key")
    print("Set OPENAI_API_KEY to try the interactive assistant!")

print("\n" + "="*80)
print("DEMO COMPLETE!")
print("="*80)
print("\nKey Takeaways:")
print("1. RAG pipeline: Retrieve → Inject Context → Generate")
print("2. Strict prompt engineering prevents hallucinations")
print("3. Always return source documents for verification")
print("4. Implement fallbacks for low-confidence matches")
print("5. Temperature=0 for deterministic, grounded answers")
print("\nNext: Hour 4 - Evaluation & Metrics")
