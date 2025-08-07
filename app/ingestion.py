import os
import tempfile
import pdfplumber
import pytesseract
from PIL import Image
from docx import Document
from fastapi import APIRouter, UploadFile, File
from app.embedding import get_embedding
import faiss
import pickle
import numpy as np

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

INDEX_PATH = "faiss_index.bin"
METADATA_PATH = "metadata.pkl"

router = APIRouter()

@router.post("/ingest")
async def ingest_file(file: UploadFile = File(...)):
    try:
        result = await handle_upload(file)
        return result
    except Exception as e:
        return {"error": str(e)}

def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
    return text.strip()

def extract_text_from_docx(file_path):
    doc = Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def extract_text_from_image(file_path):
    image = Image.open(file_path)
    return pytesseract.image_to_string(image)

def chunk_text(text):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def build_faiss_index(chunks, filename):
    embeddings = []
    new_metadata = []

    for i, chunk in enumerate(chunks):
        vector = get_embedding(chunk)
        embeddings.append(vector)
        new_metadata.append({
            "text": chunk,
            "metadata": {
                "filename": filename,
                "chunk_index": i
            }
        })

    embeddings_np = np.array(embeddings).astype('float32')

    # Always create a NEW index (fresh start every time)
    dim = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings_np)

    # Overwrite both index and metadata files
    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(new_metadata, f)

    return {"chunks_indexed": len(chunks)}


async def handle_upload(file: UploadFile):
    ext = file.filename.split(".")[-1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        if ext == "pdf":
            text = extract_text_from_pdf(tmp_path)
        elif ext == "docx":
            text = extract_text_from_docx(tmp_path)
        elif ext == "txt":
            text = extract_text_from_txt(tmp_path)
        elif ext in ["jpg", "jpeg", "png"]:
            text = extract_text_from_image(tmp_path)
        else:
            return {"error": f"Unsupported file type: .{ext}"}
    except Exception as e:
        return {"error": str(e)}
    finally:
        os.remove(tmp_path)

    chunks = chunk_text(text)
    build_result = build_faiss_index(chunks, file.filename)

    return {
        "filename": file.filename,
        "text_preview": text[:1000] + "..." if len(text) > 1000 else text,
        "chunks_created": build_result["chunks_indexed"],
        "message": "File uploaded and indexed successfully."
    }
