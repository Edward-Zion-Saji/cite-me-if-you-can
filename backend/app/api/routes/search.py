from fastapi import APIRouter, HTTPException, Depends
from typing import List

from ...models.schemas import SimilaritySearchRequest, SearchResult, JournalResponse
from ...dependencies import get_search_service
from ...services.search_service import SearchService

router = APIRouter()



@router.post("/similarity_search", response_model=List[SearchResult], tags=["search"])
async def similarity_search(
    request: SimilaritySearchRequest,
    search_service: SearchService = Depends(get_search_service)
):
    """
    Perform semantic similarity search
    
    Args:
        query: The search query
        k: Number of results to return (1-50, default: 10)
        min_score: Minimum similarity score (0.0-1.0, default: 0.25)
    """
    return await search_service.similarity_search(
        query=request.query,
        k=request.k,
        min_score=request.min_score
    )

@router.get("/{journal_id}", response_model=JournalResponse, tags=["search"])
async def get_journal(
    journal_id: str,
    search_service: SearchService = Depends(get_search_service)
):
    """
    Get metadata and all chunks for a specific journal document
    
    Args:
        journal_id: The ID of the journal document to retrieve
    """
    return await search_service.get_journal(journal_id)
