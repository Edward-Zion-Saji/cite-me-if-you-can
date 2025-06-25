from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict, Any
from sqlalchemy.orm import Session

from ...models.citation import Citation, SessionLocal
from ...services.citation_service import get_citation_service, CitationService
# No need for CitationStats import as we're using direct dict responses

router = APIRouter(prefix="/citations", tags=["citations"])

@router.get("/stats", response_model=List[Dict[str, Any]])
async def get_citation_stats(
    limit: int = 10,
    citation_service: CitationService = Depends(get_citation_service)
):
    """
    Get citation statistics for documents
    
    Args:
        limit: Maximum number of results to return (sorted by citation count)
        
    Returns:
        List of documents with their citation counts
    """
    try:
        # Get the most cited documents
        db = SessionLocal()
        citations = db.query(Citation)\
            .order_by(Citation.citation_count.desc())\
            .limit(limit)\
            .all()
            
        return [
            {
                "id": c.id,
                "source_doc_id": c.source_doc_id,
                "title": c.title,
                "citation_count": c.citation_count
            }
            for c in citations
        ]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@router.get("/document/{doc_id}", response_model=Dict[str, Any])
async def get_document_citations(
    doc_id: str,
    citation_service: CitationService = Depends(get_citation_service)
):
    """
    Get citation count for a specific document
    
    Args:
        doc_id: Document ID to get citation count for
        
    Returns:
        Document citation information
    """
    try:
        db = SessionLocal()
        citation = db.query(Citation).filter(Citation.id == doc_id).first()
        
        if not citation:
            raise HTTPException(status_code=404, detail="Document not found")
            
        return {
            "id": citation.id,
            "source_doc_id": citation.source_doc_id,
            "title": citation.title,
            "citation_count": citation.citation_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()
