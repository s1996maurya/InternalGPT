import os
import glob
import pickle
import faiss
import numpy as np
import pandas as pd
from docx import Document
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

DB_PATH = "./faiss_index"
INDEX_FILE = os.path.join(DB_PATH, "docs.index")
META_FILE = os.path.join(DB_PATH, "meta.pkl")
os.makedirs(DB_PATH, exist_ok=True)

hf_model = SentenceTransformer("all-MiniLM-L6-v2")
dimension = hf_model.get_sentence_embedding_dimension()

if os.path.exists(INDEX_FILE):
    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "rb") as f:
        docs, meta = pickle.load(f)
else:
    index = faiss.IndexFlatL2(dimension)
    docs, meta = [], []

def chunk_text(text, size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunk = " ".join(words[i:i + size])
        chunks.append(chunk)
    return chunks

def embed_and_store(chunks, source):
    vectors = hf_model.encode(chunks)
    index.add(np.array(vectors, dtype="float32"))
    docs.extend(chunks)
    meta.extend([source] * len(chunks))

def ingest_file(file):
    ext = file.split(".")[-1].lower()
    text_data = []
    if ext == "txt":
        with open(file, "r", encoding="utf-8") as f:
            text_data.append((file, f.read()))
    elif ext == "csv":
        df = pd.read_csv(file)
        text_data.append((file, df.to_string(index=False)))
    elif ext == "xlsx":
        df = pd.read_excel(file, sheet_name=None)
        for sheet, sdf in df.items():
            text_data.append((f"{file}::{sheet}", sdf.to_string(index=False)))
    elif ext in ["doc", "docx"]:
        doc = Document(file)
        text_data.append((file, "\n".join([p.text for p in doc.paragraphs if p.text.strip()])))
    elif ext == "pdf":
        reader = PdfReader(file)
        text_data.append((file, "\n".join([page.extract_text() or "" for page in reader.pages])))

    for fname, text in text_data:
        chunks = chunk_text(text)
        embed_and_store(chunks, fname)
        print(f"✅ Ingested {len(chunks)} chunks from {fname}")

if __name__ == "__main__":
    for file in glob.glob("./docs/*"):
        ingest_file(file)

    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "wb") as f:
        pickle.dump((docs, meta), f)

    print("✅ Ingestion complete. FAISS index saved.")
