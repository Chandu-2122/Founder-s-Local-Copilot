# nodes.py
"""
Defines the different functional nodes for our Founder's Desk assistant.
Each node is just a Python function we can call in the LangGraph graph.
"""

from rag import build_qa_chain, load_index

print("[nodes.py] Initializing nodes...")

VECTORSTORE = None
QA_CHAIN = None

def init_vectorstore(index_dir="vector_index"):
    global VECTORSTORE, QA_CHAIN
    if VECTORSTORE is None:
        VECTORSTORE = load_index(index_dir)
        QA_CHAIN = build_qa_chain(VECTORSTORE)
    return QA_CHAIN

def docs_node(state: dict) -> dict:
    """Answer based on startup docs (strict RAG)."""
    question = state.question
    print(f"[nodes.py][docs_node] Question: {question}")

    chat_history = []

    result = QA_CHAIN.invoke({"question": question, "chat_history": chat_history})
    answer = result["answer"]  
    return {"answer": answer, "source": "docs"}

def advice_node(state: dict) -> dict:
    """Give general startup advice (not doc based)."""
    question = state.question
    print(f"[nodes.py][advice_node] Question: {question}")
    # For simplicity, we just prepend guidance
    answer = (
        "This is general startup advice, not from your internal docs:\n"
        f"- {question}\n"
        "→ Focus on clarity, test ideas early, and track metrics."
    )
    return {"answer": answer, "source": "advice"}

def marketing_node(state: dict) -> dict:
    """Help with product/marketing copy tasks."""
    question = state.question
    print(f"[nodes.py][marketing_node] Question: {question}")
    # For demo, just return a canned template
    answer = (
        "Here’s a draft marketing response based on your request:\n"
        f"'{question}' → Highlight value, keep it simple, and show customer impact."
    )
    return {"answer": answer, "source": "marketing"}

print("[nodes.py] Nodes ready.")
