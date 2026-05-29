import uuid

from customer_support_bot.agent.graph import AGENT


def run_query(question: str, tenant_id: str = "acme") -> dict:
    initial_state = {
        "question": question,
        "session_id": str(uuid.uuid4()),
        "tenant_id": tenant_id,
        "query_embedding": None,
        "retrieved_chunks": None,
        "graded_chunks": None,
        "generation": None,
        "context_used": None,
        "faithfulness_score": None,
        "unsupported_claims": None,
        "final_answer": None,
        "citations": None,
        "route_decision": None,
        "error": None,
    }
    return AGENT.invoke(initial_state)


def main():
    print("Customer Support RAG Agent")
    print("Before first use, build the FAISS index:")
    print("python -m customer_support_bot.ingestion.ingest --tenant-id acme --filepath sample_docs/acme_sample_policy.docx")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Bye.")
            break
        state = run_query(question)
        print("\nAnswer:")
        print(state.get("final_answer", "No answer generated."))
        citations = state.get("citations") or []
        if citations:
            print("Sources:")
            for citation in citations:
                print(f"- {citation}")
        print()


if __name__ == "__main__":
    main()
