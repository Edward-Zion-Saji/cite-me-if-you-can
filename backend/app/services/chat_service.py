from typing import List, Dict, Any, Tuple
from fastapi import HTTPException
import logging
import re

from ..models.schemas import SearchResult, ChatRequest
from ..core.llm import LLMService
from ..services.search_service import SearchService

logger = logging.getLogger(__name__)

class ChatService:
    def __init__(self, llm_service: LLMService, search_service: SearchService):
        self.llm_service = llm_service
        self.search_service = search_service
    
    async def chat(self, chat_request: ChatRequest) -> Dict[str, Any]:
        """
        Generate a response to a chat query using RAG
        
        Args:
            chat_request: The chat request containing the user's query
            
        Returns:
            Dictionary containing the response and sources
        """
        try:
            if not chat_request.query or not chat_request.query.strip():
                raise HTTPException(status_code=400, detail="Query cannot be empty")
                
            # Retrieve relevant context
            results = await self.search_service.similarity_search(
                query=chat_request.query,
                k=5,  # Default to 5 results
                min_score=0.25
            )
            
            if not results:
                return {
                    "response": "No relevant information found in the knowledge base.",
                    "sources": []
                }
            
            # Build context with citations
            context_parts = []
            sources = []
            for i, res in enumerate(results, 1):
                source_info = {
                    "id": i, 
                    "source": res.source, 
                    "text": res.text,
                    "section": res.section,
                    "citation_count": res.citation_count,
                    "url": res.url if hasattr(res, 'url') else "#"  # Add URL if available
                }
                # Format source with markdown link
                source_link = f"[{res.source}]({source_info['url']})" if source_info['url'] != "#" else res.source
                context_parts.append(f"[Source {i}: {source_link}]\n{source_info['text']}")
                sources.append(source_info)
            
            context = "\n\n".join(context_parts)
            
            # Generate response with citations
            response_text = self.llm_service.generate_response([
                ("system", 
                 "You are a research assistant. Based on the query you are given the following context from appropriate sources. "
                 "Include inline citations in markdown format like [1](url) where the number links to the source. "
                 "The citation number should be a hyperlink to the original source. "
                 "When citing, use the format: [source title](source_url) where the title is clickable. "
                 "Be concise and relevant and answer the question directly without any preface. If the context doesn't contain relevant information, say: I couldn\'t find relevant information in the knowledge base."),
                ("human", f"Context:\n{context}\n\nQuestion: {chat_request.query}")
            ])
            response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
            return {
                "response": response_text,
                "sources": sources,
                "query": chat_request.query
            }
            
        except HTTPException as he:
            raise he
        except Exception as e:
            logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail="An error occurred while processing your request")
