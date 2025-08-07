import base64
from typing import Optional
from PIL import Image
import io
import pytesseract
import os
from fastapi import APIRouter
from pydantic import BaseModel


from app.embedding import get_embedding
from app.utils import faiss_search


router = APIRouter()

class QueryRequest(BaseModel):
    question: str
    image_base64: Optional[str] = None

@router.post("/query")
def query_endpoint(request: QueryRequest):
    return handle_query(request.question, request.image_base64)

def handle_query(question: str, image_base64: Optional[str] = None) -> dict:
    ocr_text = ""
    if image_base64:
        ocr_text = perform_ocr(image_base64)

    combined_query = question
    if ocr_text:
        combined_query += "\n\nAdditional context from image OCR:\n" + ocr_text

    query_vector = get_embedding(combined_query)
    retrieved_docs = faiss_search(query_vector, top_k=5)

    answer = "\n---\n".join([doc['text'] for doc in retrieved_docs])

    return {
        "answer": answer,
        "sources": [doc.get('metadata', {}) for doc in retrieved_docs]
    }



def perform_ocr(image_base64: str) -> str:
    image_bytes = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_bytes))
    text = pytesseract.image_to_string(image)
    return text.strip()
