RAG-based Smart Document QA API

A smart Retrieval-Augmented Generation (RAG) API built using FastAPI that allows users to:
- Upload any document (PDF, DOCX, TXT, image, CSV, DB, etc.)
- Extract relevant content (including OCR for images)
- Ask questions about the content via natural language
- Get accurate responses leveraging semantic search with FAISS and embeddings


## Features

-  Supports `.pdf`, `.docx`, `.txt`, `.jpg`, `.png`
-  OCR for scanned images and documents (via pytesseract)
-  Embeddings using SentenceTransformers
-  Vector search using FAISS
-  Image-based questions supported via base64
-  Modular, async-ready FastAPI backend
-  Ready for frontend or Streamlit integration


## How to Run (Instructions)

1. **Clone the repo or unzip**
   ```bash
   git clone https://github.com/nabilarahman01/excel-rag-api.git
   cd rag-api
2. Install dependencies
    ```bash
    pip install -r requirements.txt
3. Set up environtment variables (see below)
4. Run the FastAPI app
    ```bash
   uvicorn main:app --reload
5. Access the docs at:
    http://127.0.0.1:8000/docs

## Environment Setup
- Python 3.9+
- Install dependencies:
  ```bash
     pip install -r requirements.txt


@Author
Nabila Rahman
nabilarahman359@gmail.com
— for RAG API Developer recruitment @ Excel Technologies
  
