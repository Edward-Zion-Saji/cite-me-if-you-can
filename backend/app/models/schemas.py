from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Dict, Any
from fastapi import UploadFile

class JournalChunkBase(BaseModel):
    id: str
    source_doc_id: str
    chunk_index: int
    section_heading: str
    journal: str
    publish_year: int
    usage_count: int
    attributes: List[str]
    link: str
    text: str
    doi: Optional[str] = None

class UploadRequest(BaseModel):
    file: Optional[UploadFile] = None
    file_url: Optional[str] = None
    schema_version: str = "1.0"

    class Config:
        arbitrary_types_allowed = True

class SimilaritySearchRequest(BaseModel):
    query: str
    k: int = Field(default=10, ge=1, le=50)
    min_score: float = Field(default=0.25, ge=0.0, le=1.0)

class SearchResult(BaseModel):
    id: str
    doc_id: str
    score: float
    text: str
    source: str
    section: str
    citation_count: int
    usage_count: int = 0

class JournalMetadata(BaseModel):
    journal_id: str
    title: str
    journal: str
    publish_year: int
    doi: Optional[str] = None
    link: str
    total_chunks: int

class JournalChunkResponse(BaseModel):
    id: str
    text: str
    section: str
    attributes: List[str]
    citation_count: int

class JournalResponse(BaseModel):
    metadata: JournalMetadata
    chunks: List[JournalChunkResponse]

class ChatRequest(BaseModel):
    query: str
