# rag.py
"""
Very simple RAG helper: load docs, split them, make embeddings, build a FAISS index.
"""

import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain, LLMChain


from model import LLM

def ingest_folder(folder: str, index_dir: str = "vector_index"):
    print("[rag.py] Starting ingestion for:", folder)

    docs = []
    for fname in os.listdir(folder):
        if fname.endswith(".txt"):
            path = os.path.join(folder, fname)
            print(f"[rag.py] Loading {path}")
            loader = TextLoader(path, encoding="utf8")
            docs.extend(loader.load())

    print(f"[rag.py] Loaded {len(docs)} docs")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)
    print(f"[rag.py] Split into {len(split_docs)} chunks")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    vectorstore.save_local(index_dir)
    print(f"[rag.py] Saved FAISS index to {index_dir}")

def load_index(index_dir: str = "vector_index"):
    print("[rag.py] Loading FAISS index from:", index_dir)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)

def build_qa_chain(vectorstore):
    system_prompt = """You are a strict assistant that ONLY answers from the provided documents.
    If the answer is not in the documents, say: 'I don’t have information about that in the docs.'
    Do not make up facts or advice unless explicitly asked for advice."""
    
    # Question rephrasing prompt
    condense_prompt = PromptTemplate.from_template(
        "Given the following conversation and a follow-up question, rephrase the question to be standalone.\n\n{question}"
    )

    # Main prompt for answering using retrieved docs
    qa_prompt = PromptTemplate.from_template(
        system_prompt + "\n\nContext:\n{context}\n\nQuestion:\n{question}"
    )

    # LLM
    llm = LLM

    # Chain to rephrase follow-up questions
    question_generator = LLMChain(llm=llm, prompt=condense_prompt)

    # Chain to answer questions using retrieved docs
    doc_chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=qa_prompt)

    # Build final conversational retrieval chain
    qa_chain = ConversationalRetrievalChain(
        retriever=vectorstore.as_retriever(),
        combine_docs_chain=doc_chain,
        question_generator=question_generator,
        return_source_documents=True,
    )
    return qa_chain

if __name__ == "__main__":
    # quick test
    ingest_folder("sample_docs")
    vs = load_index()
    qa = build_qa_chain(vs)
    result = qa.run("What is in the docs?")
    print("[rag.py] Test QA Result:", result)
