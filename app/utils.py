import faiss
import numpy as np
import pickle
import os

INDEX_PATH = "faiss_index.bin"
METADATA_PATH = "metadata.pkl"

faiss_index = None
metadata = []

if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
    try:
        faiss_index = faiss.read_index(INDEX_PATH)
        with open(METADATA_PATH, "rb") as f:
            metadata = pickle.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load FAISS index or metadata: {e}")
else:
    print("[INFO] FAISS index or metadata not found — will be generated after ingestion.")

def faiss_search(query_vector: list[float], top_k: int = 5) -> list[dict]:
    if not os.path.exists(INDEX_PATH) or not os.path.exists(METADATA_PATH):
        raise RuntimeError("FAISS index or metadata not found. Please run ingestion first.")

    # Reload fresh every time (because index might change)
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)

    query_np = np.array(query_vector).astype('float32').reshape(1, -1)
    distances, indices = index.search(query_np, top_k)

    results = []
    for idx in indices[0]:
        if idx < len(metadata):
            results.append(metadata[idx])

    return results

