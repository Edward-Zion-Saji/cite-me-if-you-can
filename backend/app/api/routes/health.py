from fastapi import APIRouter, Depends
from typing import Dict, Any

from ...dependencies import get_vector_store
from ...core.vector_store import VectorStoreService

router = APIRouter()

@router.get("/health", tags=["health"])
async def health_check(
    vector_store: VectorStoreService = Depends(get_vector_store)
) -> Dict[str, Any]:
    """
    Health check endpoint
    
    Returns:
        Dictionary containing service status information
    """
    try:
        collection_info = vector_store.get_collection_info()
        return {
            "status": "healthy",
            "vector_store": True,
            "collection": vector_store.collection_name,
            "total_chunks": collection_info.points_count if collection_info else 0
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "vector_store": False
        }
