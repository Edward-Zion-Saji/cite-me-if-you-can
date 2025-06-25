import json
import logging
from typing import List, Dict, Any, Tuple
import httpx
from fastapi import HTTPException, UploadFile
from langchain_core.documents import Document

from ..models.schemas import JournalChunkBase

logger = logging.getLogger(__name__)

class UploadService:
    @staticmethod
    async def fetch_chunks_from_url(file_url: str) -> List[Dict[str, Any]]:
        """Fetch and parse journal chunks from a URL"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(file_url)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching from URL: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to fetch from URL: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            raise HTTPException(status_code=400, detail="Invalid JSON format in file")
        except Exception as e:
            logger.error(f"Error fetching chunks: {e}")
            raise HTTPException(status_code=500, detail=f"Error processing file: {e}")

    @staticmethod
    async def process_uploaded_file(file: UploadFile) -> List[Dict[str, Any]]:
        """Process uploaded file and extract chunks"""
        try:
            content = await file.read()
            return json.loads(content.decode('utf-8'))
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            raise HTTPException(status_code=400, detail="Invalid JSON format in uploaded file")
        except Exception as e:
            logger.error(f"Error processing uploaded file: {e}")
            raise HTTPException(status_code=500, detail=f"Error processing file: {e}")

    @staticmethod
    def chunks_to_documents(chunks_data: List[Dict[str, Any]]) -> Tuple[List[Document], Dict[str, int]]:
        """Convert chunk data to LangChain Documents and track journal citations"""
        documents = []
        journal_citations = {}  # Track citation count per journal
        
        # First pass: collect all source documents and initialize citation counts
        for chunk_data in chunks_data:
            chunk = JournalChunkBase(**chunk_data)
            if chunk.source_doc_id not in journal_citations:
                journal_citations[chunk.source_doc_id] = 0
        
        # Second pass: create documents with journal-level citation count
        for chunk_data in chunks_data:
            chunk = JournalChunkBase(**chunk_data)
            doc = Document(
                page_content=chunk.text,
                metadata={
                    "id": chunk.id,
                    "source_doc_id": chunk.source_doc_id,
                    "chunk_index": chunk.chunk_index,
                    "section_heading": chunk.section_heading,
                    "journal": chunk.journal,
                    "publish_year": chunk.publish_year,
                    "usage_count": chunk.usage_count,
                    "attributes": chunk.attributes,
                    "link": chunk.link,
                    **({"doi": chunk.doi} if chunk.doi else {})
                }
            )
            documents.append(doc)
        
        return documents, journal_citations
