from fastapi import Depends
from .core.vector_store import VectorStoreService
from .core.embeddings import EmbeddingService
from .core.llm import LLMService
from .services.upload_service import UploadService
from .services.search_service import SearchService
from .services.chat_service import ChatService
from .services.citation_service import get_citation_service, CitationService

def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()

def get_vector_store(embedding_service: EmbeddingService = Depends(get_embedding_service)) -> VectorStoreService:
    vector_store = VectorStoreService()
    vector_store.initialize(embedding_service.embeddings)
    return vector_store

def get_llm_service() -> LLMService:
    return LLMService()

def get_upload_service() -> UploadService:
    return UploadService()

def get_search_service(
    vector_store: VectorStoreService = Depends(get_vector_store),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    citation_service: CitationService = Depends(get_citation_service)
) -> SearchService:
    return SearchService(vector_store, embedding_service, citation_service)

def get_chat_service(
    llm_service: LLMService = Depends(get_llm_service),
    search_service: SearchService = Depends(get_search_service),
    citation_service: CitationService = Depends(get_citation_service)
) -> ChatService:
    return ChatService(llm_service, search_service, citation_service)
