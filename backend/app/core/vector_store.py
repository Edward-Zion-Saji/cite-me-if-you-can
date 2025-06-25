from typing import List, Dict, Any, Optional, Tuple
import uuid
import logging
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore

from ..config import settings

logger = logging.getLogger(__name__)

class VectorStoreService:
    def __init__(self):
        self.client = QdrantClient(
            url=settings.QDRANT_URL, 
            prefer_grpc=settings.QDRANT_PREFER_GRPC
        )
        self.collection_name = settings.QDRANT_COLLECTION
        self.vector_store = None
        
    def initialize(self, embeddings):
        """Initialize the vector store with embeddings"""
        try:
            self.vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=embeddings,
                metadata_payload_key="metadata"
            )
            logger.info("Vector store initialized successfully")
            return self.vector_store
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise
    
    def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to the vector store"""
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
            
        try:
            # Generate UUIDs for each document
            ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.metadata["id"])) for doc in documents]
            self.vector_store.add_documents(documents, ids=ids)
            return True
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
    
    def search(self, query_embedding: List[float], k: int = 10, min_score: float = 0.25):
        """Search the vector store"""
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
            
        return self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=k,
            score_threshold=min_score,
            with_payload=True
        )
    
    def get_journal_chunks(self, journal_id: str, limit: int = 1000):
        """Get all chunks for a specific journal"""
        records, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(
                    key="metadata.source_doc_id",
                    match=models.MatchValue(value=journal_id)
                )]
            ),
            with_payload=True,
            limit=limit
        )
        return sorted(records, key=lambda x: x.payload["metadata"]["chunk_index"])
    
    def get_collection_info(self):
        """Get collection information"""
        try:
            return self.client.get_collection(self.collection_name)
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return None
