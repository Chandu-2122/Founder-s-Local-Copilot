# Founder-s-Local-Copilot

## Objective
A private AI copilot that answers only from startup’s own documents (policies, roadmaps, investor memos).

## Tech & Requirements
- Streamlit
- LangChain, LangChain Community, LangChain Ollama
- LangGraph (for workflow routing)
- FAISS + Sentence Transformers (for embeddings)
- Unstructured (for local doc ingestion)

## Project Files
- **app.py** → Streamlit app
- **graph.py** → defines app flow with LangGraph
- **nodes.py** → nodes for docs, advice, marketing
- **rag.py** → handles RAG ingestion, FAISS indexing
- **model.py** → LLM connection.
- **utils.py** → helpers

## Challenges Faced
- Handling FAISS index load/save correctly.
- Preventing the AI from hallucinating outside of documents.

## Learnings
- RAG improves accuracy but depends on good chunking & embeddings.
- Routing with LangGraph makes the system modular and expandable.

## Future Improvements
- Add multi-doc support (PDFs, slides).
- Implement caching for faster retrieval.
- Expand beyond docs to handle integrations (email, Slack).
  



   
       
