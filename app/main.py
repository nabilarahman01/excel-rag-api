from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.ingestion import router as ingestion_router
from app.query import router as query_router

app = FastAPI(title="RAG API for Excel Technologies")

# Include Routers
app.include_router(ingestion_router, tags=["Ingestion"])
app.include_router(query_router, tags=["Query"])

# CORS Middleware (optional for local dev or frontend integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "RAG API is live"}
