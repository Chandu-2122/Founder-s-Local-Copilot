# app.py
"""
Streamlit app to interact with the Founder's Desk assistant.
"""

import os
import streamlit as st
from graph import app_graph
from nodes import init_vectorstore
from rag import ingest_folder, load_index, build_qa_chain

INDEX_DIR = "vector_index"
DOCS_DIR = "sample_docs"

st.set_page_config(page_title="Founder's Local Copilot")
st.title("🧠 Founder's Local Copilot")

# Sidebar doc preview
with open(os.path.join(DOCS_DIR, "about.txt"), "r", encoding="utf-8") as f:
    st.sidebar.header("Reference Document")
    st.sidebar.text_area("Doc content", f.read(), height=300)

# Ensure index exists
if not os.path.exists(INDEX_DIR):
    st.info("No vector index found. Building index from sample_docs...")
    ingest_folder(DOCS_DIR, INDEX_DIR)

# Init vectorstore + QA chain
qa_chain = init_vectorstore(INDEX_DIR)

st.write("Ask about your startup docs, get marketing help, or general advice.")

if "history" not in st.session_state:
    st.session_state.history = []

question = st.text_input("Your question:")

if st.button("Ask") and question:
    # Run through LangGraph
    state = {"question": question, "answer": "", "source": ""}
    result = app_graph.invoke(state)

    answer = result["answer"]
    source = result["source"]

    # Save to history
    st.session_state.history.append((question, answer, source))

# Show history
st.subheader("Conversation")
for q, a, src in st.session_state.history:
    st.markdown(f"**Q:** {q}")
    st.markdown(f"**A ({src})** {a}")
    st.markdown("---")
