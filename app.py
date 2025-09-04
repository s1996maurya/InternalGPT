import os
import io
import pandas as pd
from docx import Document
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import streamlit as st
import faiss
import numpy as np

# ---------- CONFIG ----------
genai.configure(api_key="AIzaSyCZSmQvr51DLq213uUTznSLKh2M7T7PNy4")
gemini_model = genai.GenerativeModel("gemini-2.0-flash")
hf_model = SentenceTransformer("all-MiniLM-L6-v2")

# FAISS in-memory index
dimension = hf_model.get_sentence_embedding_dimension()
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = faiss.IndexFlatL2(dimension)
    st.session_state.docs = []
    st.session_state.meta = []

# ---------- HELPERS ----------
def chunk_text(text, size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunk = " ".join(words[i:i + size])
        chunks.append(chunk)
    return chunks

def embed_and_store(chunks, source):
    vectors = hf_model.encode(chunks)
    st.session_state.faiss_index.add(np.array(vectors, dtype="float32"))
    st.session_state.docs.extend(chunks)
    st.session_state.meta.extend([source] * len(chunks))

def ingest_file(file):
    ext = file.name.split(".")[-1].lower()
    text_data = []

    if ext == "txt":
        text_data.append((file.name, file.read().decode("utf-8")))
    elif ext == "csv":
        df = pd.read_csv(file)
        text_data.append((file.name, df.to_string(index=False)))
    elif ext == "xlsx":
        df = pd.read_excel(file, sheet_name=None)
        for sheet, sdf in df.items():
            text_data.append((f"{file.name}::{sheet}", sdf.to_string(index=False)))
    elif ext in ["doc", "docx"]:
        doc = Document(io.BytesIO(file.read()))
        text_data.append((file.name, "\n".join([p.text for p in doc.paragraphs if p.text.strip()])))
    elif ext == "pdf":
        reader = PdfReader(io.BytesIO(file.read()))
        text_data.append((file.name, "\n".join([page.extract_text() or "" for page in reader.pages])))
    else:
        st.warning(f"Unsupported file type: {ext}")
        return

    for fname, text in text_data:
        chunks = chunk_text(text)
        embed_and_store(chunks, fname)
        st.success(f"✅ Ingested {len(chunks)} chunks from {fname}")

def search(query, top_k=3):
    if st.session_state.faiss_index.ntotal == 0:
        return [], []
    q_vec = hf_model.encode([query]).astype("float32")
    D, I = st.session_state.faiss_index.search(q_vec, top_k)
    docs = [st.session_state.docs[i] for i in I[0]]
    sources = [st.session_state.meta[i] for i in I[0]]
    return docs, sources

# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="Internal GPT Assistant", page_icon="🤖")
st.title("🤖 Internal GPT Assistant (FAISS Edition)")

uploaded_files = st.file_uploader(
    "Upload PRDs, Configs, Bug Reports (TXT, CSV, XLSX, DOCX, PDF)",
    type=["txt", "csv", "xlsx", "docx", "doc", "pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    for f in uploaded_files:
        ingest_file(f)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.chat_input("Ask me about PRDs, configs, or bugs...")

if query:
    docs, sources = search(query, top_k=3)
    context = "\n\n".join(docs)

    response = gemini_model.generate_content(
        f"Answer the question using only the context.\n\nQuestion: {query}\n\nContext:\n{context}"
    )
    answer = response.text
    st.session_state.chat_history.append((query, answer, sources))

for q, a, s in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(a)
        with st.expander("📂 Sources"):
            st.write(s)
