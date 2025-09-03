# app.py
import os
import io
import chromadb
import pandas as pd
from docx import Document
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import streamlit as st

# ---------- CONFIG ----------
DB_PATH = "./chroma_db"
COLLECTION_NAME = "prd_knowledge"
genai.configure(api_key="AIzaSyCZSmQvr51DLq213uUTznSLKh2M7T7PNy4")

# HuggingFace embeddings wrapper
class HFEmbeddingFunction:
    def __init__(self, model):
        self.model = model
    def __call__(self, input: list[str]) -> list[list[float]]:
        return self.model.encode(input).tolist()
    def name(self): return "hf-sentence-transformers"

hf_model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_function = HFEmbeddingFunction(hf_model)

# Init Chroma
client = chromadb.Client()
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_function
)

# Init Gemini
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# ---------- HELPERS ----------
def chunk_text(text, size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunk = " ".join(words[i:i + size])
        chunks.append(chunk)
    return chunks

def ingest_file(file):
    """Parse uploaded file and store chunks in ChromaDB"""
    ext = file.name.split(".")[-1].lower()
    text_data = []

    if ext == "txt":
        text = file.read().decode("utf-8")
        text_data.append((file.name, text))

    elif ext == "csv":
        df = pd.read_csv(file)
        text_data.append((file.name, df.to_string(index=False)))

    elif ext == "xlsx":
        df = pd.read_excel(file, sheet_name=None)
        for sheet, sdf in df.items():
            text_data.append((f"{file.name}::{sheet}", sdf.to_string(index=False)))

    elif ext in ["doc", "docx"]:
        doc = Document(io.BytesIO(file.read()))
        text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        text_data.append((file.name, text))

    elif ext == "pdf":
        reader = PdfReader(io.BytesIO(file.read()))
        text = "\n".join([page.extract_text() or "" for page in reader.pages])
        text_data.append((file.name, text))

    else:
        st.warning(f"Unsupported file type: {ext}")
        return

    # Chunk + store
    ids, texts, metas = [], [], []
    for fname, text in text_data:
        for i, chunk in enumerate(chunk_text(text)):
            ids.append(f"{fname}_{i}")
            texts.append(chunk)
            metas.append({"source": fname})

    if texts:
        collection.add(documents=texts, ids=ids, metadatas=metas)
        st.success(f"✅ Ingested {len(texts)} chunks from {file.name}")

# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="Internal GPT Assistant", page_icon="🤖")
st.title("🤖 Internal GPT Assistant")

# File upload
uploaded_files = st.file_uploader(
    "Upload PRDs, Configs, Bug Reports (TXT, CSV, XLSX, DOCX, PDF)", 
    type=["txt", "csv", "xlsx", "docx", "doc", "pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    for f in uploaded_files:
        ingest_file(f)

# Chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.chat_input("Ask me about PRDs, configs, or bugs...")

if query:
    results = collection.query(query_texts=[query], n_results=3)
    docs = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    context = "\n\n".join(docs)

    response = gemini_model.generate_content(
        f"Answer the question using only the context.\n\nQuestion: {query}\n\nContext:\n{context}"
    )
    answer = response.text

    st.session_state.chat_history.append((query, answer, sources))

# Display chat history
for q, a, s in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(a)
        with st.expander("📂 Sources"):
            st.write(s)
