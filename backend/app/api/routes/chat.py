from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any

from ...models.schemas import ChatRequest
from ...dependencies import get_chat_service
from ...services.chat_service import ChatService

router = APIRouter()

@router.post("", tags=["chat"])
async def chat_with_llm(
    chat_request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service)
) -> Dict[str, Any]:
    """
    Chat with the LLM using RAG (Retrieval-Augmented Generation)
    
    This endpoint takes a user query, retrieves relevant context from the knowledge base,
    and generates a response using the LLM.
    
    Args:
        query: The user's query
        
    Returns:
        A dictionary containing the response and sources used
    """
    return await chat_service.chat(chat_request)
