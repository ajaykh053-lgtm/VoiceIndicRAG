"""Health check & diagnostics router."""

from fastapi import APIRouter
from app.core.config import settings
from app.services.vector_store import vector_store

router = APIRouter()


@router.get("/")
async def health_check():
    """Health check returning system and index status."""
    return {
        "status": "healthy",
        "service": "Multilingual Voice RAG API",
        "faiss_total_chunks": vector_store.total_count,
        "embedding_model": settings.EMBEDDING_MODEL,
        "groq_model": settings.GROQ_MODEL,
        "groq_configured": bool(settings.GROQ_API_KEY and len(settings.GROQ_API_KEY) > 5),
        "sarvam_configured": bool(settings.SARVAM_API_KEY and len(settings.SARVAM_API_KEY) > 5),
    }
