from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from ..models.citation import Citation, SessionLocal
from ..models.schemas import SearchResult
import logging

logger = logging.getLogger(__name__)

class CitationService:
    def __init__(self, db: Session):
        self.db = db
    
    def increment_citation_counts(self, search_results: List[SearchResult]) -> None:
        """Increment citation counts for documents referenced in search results"""
        if not search_results:
            return
            
        try:
            # Group results by source_doc_id to avoid duplicate counts for the same document
            seen_docs = set()
            unique_results = []
            
            for result in search_results:
                # Use source as document identifier since it's the closest we have to source_doc_id
                doc_key = result.source
                if doc_key not in seen_docs:
                    seen_docs.add(doc_key)
                    unique_results.append((doc_key, result))
            
            # Update citation counts for each unique document
            for doc_key, result in unique_results:
                # Try to find existing citation record by source_doc_id
                citation = self.db.query(Citation).filter(Citation.source_doc_id == doc_key).first()
                
                if citation:
                    # Update existing citation
                    citation.citation_count += 1
                    # Update the ID to the latest chunk ID for reference
                    citation.id = result.id
                else:
                    # Create new citation record
                    citation = Citation(
                        id=result.id,
                        source_doc_id=doc_key,
                        citation_count=1,
                        title=result.section or doc_key  # Use section or doc_key as title
                    )
                    self.db.add(citation)
                
                # Update the citation count in all results from this document
                for r in search_results:
                    if r.source == doc_key:
                        r.citation_count = citation.citation_count
            
            self.db.commit()
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error updating citation counts: {str(e)}")
            # Don't fail the request if citation tracking fails
            
    def get_citation_counts(self, doc_ids: List[str]) -> Dict[str, int]:
        """Get citation counts for multiple document IDs"""
        try:
            citations = self.db.query(Citation).filter(Citation.id.in_(doc_ids)).all()
            return {citation.id: citation.citation_count for citation in citations}
        except Exception as e:
            logger.error(f"Error fetching citation counts: {str(e)}")
            return {}

# Dependency
def get_citation_service():
    db = SessionLocal()
    try:
        yield CitationService(db)
    finally:
        db.close()
