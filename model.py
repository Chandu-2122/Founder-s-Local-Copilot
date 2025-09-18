# model
from langchain_ollama.llms import OllamaLLM
print("[model.py] Initializing LLM…")

LLM = OllamaLLM(
    model="llama3.2:latest",
    base_url="http://127.0.0.1:11434"
)
print("[model.py] LLM ready:", type(LLM))
