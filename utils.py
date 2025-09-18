# utils.py
import os
from langchain.docstore.document import Document

def safe_preview(doc, length=300):
    txt = doc.page_content if hasattr(doc, "page_content") else str(doc)
    return (txt[:length] + "...") if len(txt) > length else txt

def list_sample_docs(folder="sample_docs"):
    if not os.path.exists(folder):
        return []
    return sorted([p for p in os.listdir(folder) if os.path.isfile(os.path.join(folder, p))])
