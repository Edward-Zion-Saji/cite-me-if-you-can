from typing import List, Dict, Any, Optional
from fastapi import HTTPException
import logging

from ..models.schemas import SearchResult, JournalResponse, JournalMetadata, JournalChunkResponse
from ..core.vector_store import VectorStoreService
from ..core.embeddings import EmbeddingService

logger = logging.getLogger(__name__)

class SearchService:
    def __init__(self, vector_store: VectorStoreService, embedding_service: EmbeddingService, citation_service: Optional[Any] = None):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.citation_service = citation_service
    
    async def similarity_search(self, query: str, k: int = 10, min_score: float = 0.25) -> List[SearchResult]:
        """Perform semantic similarity search and update usage counts"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_service.embed_query(query)
            
            # Perform search
            results = self.vector_store.search(query_embedding, k=k, min_score=min_score)
            
            # Format results and collect chunk IDs for usage tracking
            search_results = []
            chunk_ids = []
            doc_ids = set()
            
            # First pass: extract doc_ids and prepare initial results
            for result in results:
                metadata = result.payload["metadata"]
                doc_id = metadata["source_doc_id"]
                doc_ids.add(doc_id)
                chunk_id = str(result.id)
                
                search_results.append({
                    'id': chunk_id,
                    'score': result.score,
                    'doc_id': doc_id,
                    'text': result.payload["page_content"],
                    'source': metadata["link"],
                    'section': metadata["section_heading"],
                    'usage_count': metadata.get("usage_count", 0),
                    'citation_count': 0  # Will be updated after fetching from citation service
                })
                chunk_ids.append(chunk_id)
            
            # Fetch citation counts for all chunk IDs
            citation_counts = {}
            if self.citation_service and chunk_ids:
                citation_counts = self.citation_service.get_citation_counts(chunk_ids)
            
            # Update search results with citation counts
            for result in search_results:
                result['citation_count'] = citation_counts.get(result['id'], 0)
            
            # Convert to SearchResult objects
            search_results = [
                SearchResult(
                    id=result['id'],
                    score=result['score'],
                    doc_id=result['doc_id'],
                    text=result['text'],
                    source=result['source'],
                    section=result['section'],
                    citation_count=result['citation_count'],
                    usage_count=result['usage_count']
                )
                for result in search_results
            ]
            
            # Update usage counts for retrieved chunks (fire and forget)
            if chunk_ids:
                self.vector_store.increment_usage_count(chunk_ids)
            
            logger.info(f"Similarity search returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"Similarity search failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
    
    async def get_journal(self, journal_id: str) -> JournalResponse:
        """Get metadata and all chunks for a specific journal document"""
        try:
            # Get all chunks for this journal
            records = self.vector_store.get_journal_chunks(journal_id)
            
            if not records:
                raise HTTPException(status_code=404, detail="Journal not found")
            
            # Extract metadata from first chunk (should be consistent across chunks)
            first_chunk = records[0].payload["metadata"]
            
            # Build metadata
            metadata = JournalMetadata(
                journal_id=journal_id,
                title=first_chunk.get("section_heading", "Unknown Title"),
                journal=first_chunk["journal"],
                publish_year=first_chunk["publish_year"],
                doi=first_chunk.get("doi"),
                link=first_chunk["link"],
                total_chunks=len(records)
            )
            
            # Build chunk responses
            chunks = []
            for record in records:
                chunk_metadata = record.payload["metadata"]
                chunks.append(JournalChunkResponse(
                    id=chunk_metadata["id"],
                    text=record.payload["page_content"],
                    section=chunk_metadata["section_heading"],
                    attributes=chunk_metadata["attributes"],
                    citation_count=chunk_metadata.get("citation_count", 0)
                ))
            
            return JournalResponse(
                metadata=metadata,
                chunks=chunks
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get journal {journal_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to retrieve journal: {str(e)}")
