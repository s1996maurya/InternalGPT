# query_docs_gemini.py
import os
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# ---------- CONFIG ----------
DB_PATH = "./chroma_db"
COLLECTION_NAME = "prd_knowledge"
genai.configure(api_key="AIzaSyCZSmQvr51DLq213uUTznSLKh2M7T7PNy4")

# HuggingFace embeddings wrapper for Chroma
class HFEmbeddingFunction:
    def __init__(self, model):
        self.model = model

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self.model.encode(input).tolist()

    def name(self):
        return "hf-sentence-transformers"

# HuggingFace embeddings
hf_model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_function = HFEmbeddingFunction(hf_model)

# Init Chroma
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_function
)

# Init Gemini
#genai.configure(api_key=os.environ["AIzaSyCZSmQvr51DLq213uUTznSLKh2M7T7PNy4"])
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# ---------- QUERY ----------
def query_docs(question: str, top_k: int = 3):
    # Retrieve from Chroma
    results = collection.query(query_texts=[question], n_results=top_k)
    docs = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]

    # Build context
    context = "\n\n".join(docs)

    # Ask Gemini
    response = gemini_model.generate_content(
        f"Answer the question based only on the context.\n\nQuestion: {question}\n\nContext:\n{context}"
    )

    return response.text, sources

# ---------- RUN ----------
if __name__ == "__main__":
    while True:
        q = input("\nAsk a question (or 'exit'): ")
        if q.lower() in ["exit", "quit"]:
            break

        answer, sources = query_docs(q)
        print("\n💡 Answer:", answer)
        print("\n📂 Sources:", sources)
