from typing import List, Tuple, Dict, Any
from langchain_sambanova import ChatSambaNovaCloud
from ..config import settings

class LLMService:
    def __init__(self):
        self.llm = ChatSambaNovaCloud(
            model=settings.LLM_MODEL,
            sambanova_api_key=settings.SAMBANOVA_API_KEY
        )
    
    def generate_response(self, messages: List[Tuple[str, str]], **kwargs) -> str:
        """
        Generate a response using the LLM
        
        Args:
            messages: List of (role, content) tuples
            **kwargs: Additional arguments to pass to the LLM
            
        Returns:
            Generated response text
        """
        try:
            response = self.llm.invoke(messages, **kwargs)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            raise ValueError(f"Failed to generate response: {str(e)}")
