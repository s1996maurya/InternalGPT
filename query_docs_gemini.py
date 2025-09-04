import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

DB_PATH = "./faiss_index"
INDEX_FILE = os.path.join(DB_PATH, "docs.index")
META_FILE = os.path.join(DB_PATH, "meta.pkl")

hf_model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index(INDEX_FILE)
with open(META_FILE, "rb") as f:
    docs, meta = pickle.load(f)

genai.configure(api_key="AIzaSyCZSmQvr51DLq213uUTznSLKh2M7T7PNy4")
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

def search(query, top_k=3):
    q_vec = hf_model.encode([query]).astype("float32")
    D, I = index.search(q_vec, top_k)
    return [(docs[i], meta[i]) for i in I[0]]

if __name__ == "__main__":
    while True:
        q = input("\nAsk a question (or 'exit'): ")
        if q.lower() in ["exit", "quit"]:
            break
        results = search(q, top_k=3)
        context = "\n\n".join([doc for doc, _ in results])
        response = gemini_model.generate_content(
            f"Answer the question using only the context.\n\nQuestion: {q}\n\nContext:\n{context}"
        )
        print("\n💡 Answer:", response.text)
        print("\n📂 Sources:", [src for _, src in results])
