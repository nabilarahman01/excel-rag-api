import os
import tempfile
import fitz  
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


FAISS_INDEX_PATH = "faiss_index.bin"
METADATA_PATH = "metadata.pkl"

def get_embedding(text: str) -> list[float]:
    return embedding_model.encode(text).tolist()


async def embed_and_store_document(file):
    # Save uploaded file to a temp location
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(await file.read())
        temp_filepath = temp_file.name

    # Load PDF using PyMuPDF
    doc = fitz.open(temp_filepath)

    chunks = []
    metadata = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()

        # Simple chunking by sentence
        sentences = text.split(". ")
        for i, chunk in enumerate(sentences):
            chunk = chunk.strip()
            if chunk:
                chunks.append(chunk)
                metadata.append({
                    "text": chunk,
                    "metadata": {
                        "filename": file.filename,
                        "page": page_num + 1,
                        "chunk_index": i
                    }
                })

    # Get embeddings
    embeddings = [get_embedding(chunk) for chunk in chunks]
    embeddings_np = np.array(embeddings).astype("float32")

    # Build FAISS index
    dim = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings_np)

    # Save FAISS index
    faiss.write_index(index, FAISS_INDEX_PATH)

    # Save metadata
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

    # Cleanup temp file
    os.remove(temp_filepath)

    return {"message": "Document embedded and stored successfully."}
