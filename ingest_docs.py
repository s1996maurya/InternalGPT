# ingest_docs_gemini.py (fixed)
import os
import glob
import chromadb
from sentence_transformers import SentenceTransformer
import pandas as pd
# ---------- CONFIG ----------
DB_PATH = "./chroma_db"
COLLECTION_NAME = "prd_knowledge"

# HuggingFace embedding model
hf_model = SentenceTransformer("all-MiniLM-L6-v2")

# Define embedding function wrapper
class HFEmbeddingFunction:
    def __init__(self, model):
        self.model = model

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self.model.encode(input).tolist()

    def name(self):
        return "hf-sentence-transformers"

embedding_function = HFEmbeddingFunction(hf_model)

# Init Chroma
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_function
)

# Load files (txt for demo)
'''
def load_files(folder="./docs"):
    data = []
    for file in glob.glob(f"{folder}/*.txt"):
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()
            data.append((file, text))
    return data
'''
def load_files(folder="./docs"):
    data = []
    for file in glob.glob(f"{folder}/*"):
        ext = file.split(".")[-1].lower()

        if ext == "txt":
            with open(file, "r", encoding="utf-8") as f:
                text = f.read()
                data.append((file, text))

        elif ext == "csv":
            df = pd.read_csv(file)
            # Combine all rows into one big string
            text = df.to_string(index=False)
            data.append((file, text))

        elif ext == "xlsx":
            df = pd.read_excel(file, sheet_name=None)  # load all sheets
            for sheet, sdf in df.items():
                text = sdf.to_string(index=False)
                data.append((f"{file}::{sheet}", text))

        else:
            print(f"Skipping unsupported file: {file}")
    
    return data


# Chunk text
def chunk_text(text, size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunk = " ".join(words[i:i + size])
        chunks.append(chunk)
    return chunks

# Ingest
if __name__ == "__main__":
    docs = load_files("./docs")
    ids, texts, metas = [], [], []

    for file, text in docs:
        for i, chunk in enumerate(chunk_text(text)):
            ids.append(f"{file}_{i}")
            texts.append(chunk)
            metas.append({"source": file})

    collection.add(
        documents=texts,
        ids=ids,
        metadatas=metas
    )

    print(f"✅ Ingested {len(texts)} chunks into ChromaDB")
