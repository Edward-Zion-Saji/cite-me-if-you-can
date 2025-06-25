from typing import List, Optional
from langchain_community.embeddings import FastEmbedEmbeddings
from ..config import settings

class EmbeddingService:
    def __init__(self):
        self.embeddings = FastEmbedEmbeddings(
            model_name=settings.EMBEDDING_MODEL
        )
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        return self.embeddings.embed_query(text)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        return self.embeddings.embed_documents(texts)
