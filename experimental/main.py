import os
from fastapi import FastAPI, HTTPException, File, UploadFile, Request, Depends, Body
from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional
import httpx
import logging
from langchain_core.documents import Document
from qdrant_client import QdrantClient, models
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_sambanova import ChatSambaNovaCloud

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Journal RAG API", version="1.0.0")

# Initialize components
client = QdrantClient(url="http://localhost:6333", prefer_grpc=True)
collection_name = "research_data"
embeddings = FastEmbedEmbeddings(model_name="thenlper/gte-large")

os.environ['SAMBANOVA_API_KEY'] = os.getenv('SAMBANOVA_API_KEY')
llm = ChatSambaNovaCloud(model="DeepSeek-R1")

# Initialize vector store
try:
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
        metadata_payload_key="metadata"
    )
    logger.info("Vector store initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize vector store: {e}")
    vector_store = None

# Pydantic models
class JournalChunk(BaseModel):
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
    file_url: Optional[str] = None  # Using str instead of HttpUrl for better compatibility
    schema_version: str = "1.0"

    class Config:
        arbitrary_types_allowed = True  # To allow UploadFile in Pydantic model

class SimilaritySearchRequest(BaseModel):
    query: str
    k: int = Field(default=10, ge=1, le=50)
    min_score: float = Field(default=0.25, ge=0.0, le=1.0)

class SearchResult(BaseModel):
    id: str
    score: float
    text: str
    source: str
    section: str
    citation_count: int

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

async def fetch_chunks_from_url(file_url: str) -> List[JournalChunk]:
    """Fetch and parse journal chunks from a URL"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(file_url)
            response.raise_for_status()
            
            # Parse JSON data
            data = response.json()
            
            # Convert to JournalChunk objects
            chunks = []
            for chunk_data in data:
                chunk = JournalChunk(**chunk_data)
                chunks.append(chunk)
            
            logger.info(f"Successfully fetched {len(chunks)} chunks from URL")
            return chunks
            
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error fetching from URL: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to fetch from URL: {e}")
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON format in file")
    except Exception as e:
        logger.error(f"Error fetching chunks: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")

async def process_uploaded_file(file: UploadFile) -> List[JournalChunk]:
    """Process uploaded file and extract chunks"""
    try:
        # Read file content
        content = await file.read()
        
        # Parse JSON
        data = json.loads(content.decode('utf-8'))
        
        # Convert to JournalChunk objects
        chunks = []
        for chunk_data in data:
            chunk = JournalChunk(**chunk_data)
            chunks.append(chunk)
        
        logger.info(f"Successfully processed {len(chunks)} chunks from uploaded file")
        return chunks
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON format in uploaded file")
    except Exception as e:
        logger.error(f"Error processing uploaded file: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")

def chunks_to_documents(chunks: List[JournalChunk]) -> List[Document]:
    """Convert JournalChunk objects to LangChain Documents"""
    documents = []
    journal_citations = {}  # Track citation count per journal
    
    # First pass: collect all source documents and initialize citation counts
    for chunk in chunks:
        if chunk.source_doc_id not in journal_citations:
            journal_citations[chunk.source_doc_id] = 0
    
    # Second pass: create documents with journal-level citation count
    for chunk in chunks:
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

async def get_upload_request(
    request: Request,
    file: Optional[UploadFile] = File(None, description="JSON file containing journal chunks"),
) -> UploadRequest:
    """Parse the upload request which can be either a file upload or a URL"""
    form_data = await request.form()
    file_url = form_data.get('file_url')
    
    # Handle file upload
    if file is not None:
        if file.content_type not in ["application/json", "text/plain"]:
            raise HTTPException(status_code=400, detail="Uploaded file must be JSON")
        return UploadRequest(file=file, file_url=None, schema_version="1.0")
    
    # Handle URL
    if file_url:
        try:
            return UploadRequest(
                file=None,
                file_url=file_url,
                schema_version=form_data.get('schema_version', '1.0')
            )
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid URL provided")
    
    raise HTTPException(
        status_code=400,
        detail="Either 'file' (multipart/form-data) or 'file_url' (form field) must be provided"
    )

@app.put("/api/upload", status_code=202)
async def upload_journal(
    upload_request: UploadRequest = Depends(get_upload_request),
    file: UploadFile = File(None)
):
    """
    Upload journal chunks and generate embeddings.
    
    This endpoint accepts either:
    1. A file upload with a JSON array of chunks
    2. A JSON body with 'file_url' pointing to a JSON file containing chunks
    
    The request should include a 'schema_version' field (default: "1.0")
    
    Example JSON body:
    {
        "file_url": "https://example.com/chunks.json",
        "schema_version": "1.0"
    }
    """
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not available")
    
    try:
        chunks = []
        # Get chunks from either file or URL
        if upload_request.file_url:
            chunks = await fetch_chunks_from_url(upload_request.file_url)
            logger.info(f"Fetched {len(chunks)} chunks from URL: {upload_request.file_url}")
        else:
            chunks = await process_uploaded_file(file)
            logger.info(f"Processed {len(chunks)} chunks from uploaded file")
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No valid chunks found to process")
            
        # Convert to documents and get journal citation tracking
        documents, journal_citations = chunks_to_documents(chunks)
        
        # Store journal citation counts in a persistent store (e.g., database)
        # For now, we'll just log them
        logger.info(f"Initialized citation counts for {len(journal_citations)} journals")
        
        # Generate UUIDs from chunk IDs for Qdrant compatibility
        import hashlib
        import uuid
        
        def id_to_uuid(id_str: str) -> str:
            # Create a UUID5 (name-based) using the SHA-1 hash of the ID string
            namespace = uuid.NAMESPACE_DNS  # Using DNS namespace for consistency
            return str(uuid.uuid5(namespace, id_str))
        
        # Generate consistent UUIDs for each chunk
        uuids = [id_to_uuid(chunk.id) for chunk in chunks]
        
        # In a real implementation, you might want to process this asynchronously
        # since we're returning 202 Accepted
        try:
            vector_store.add_documents(documents, ids=uuids)
            logger.info(f"Successfully indexed {len(documents)} chunks from {len(journal_citations)} journals")
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            # Even if indexing fails, we still return 202 since the request was accepted
        
        # Return 202 Accepted response
        return {
            "status": "accepted",
            "message": "Request accepted for processing",
            "chunk_count": len(documents),
            "journal_count": len(journal_citations),
            "schema_version": upload_request.schema_version
        }
    
    except HTTPException as he:
        logger.warning(f"Client error in upload_journal: {str(he.detail)}")
        raise he
    except json.JSONDecodeError as je:
        logger.error(f"JSON decode error: {str(je)}")
        raise HTTPException(status_code=400, detail=f"Invalid JSON format: {str(je)}")
    except Exception as e:
        logger.error(f"Unexpected error in upload_journal: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )

@app.post("/api/similarity_search")
async def similarity_search(request: SimilaritySearchRequest):
    """Perform semantic similarity search"""
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not available")
    
    try:
        # Generate query embedding
        query_embedding = embeddings.embed_query(request.query)
        
        # Perform search
        results = client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=request.k,
            score_threshold=request.min_score,
            with_payload=True
        )
        
        # Format results
        search_results = []
        for result in results:
            metadata = result.payload["metadata"]
            search_results.append(SearchResult(
                id=result.id,
                score=result.score,
                text=result.payload["page_content"],
                source=metadata["link"],
                section=metadata["section_heading"],
                citation_count=metadata.get("citation_count", 0)
            ))
        
        logger.info(f"Similarity search returned {len(search_results)} results")
        return search_results
    
    except Exception as e:
        logger.error(f"Similarity search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/api/{journal_id}")
async def get_journal(journal_id: str):
    """Get metadata and all chunks for a specific journal document"""
    try:
        # Query for all chunks of this journal
        records, _ = client.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(
                    key="metadata.source_doc_id",
                    match=models.MatchValue(value=journal_id)
                )]
            ),
            with_payload=True,
            limit=1000  # Adjust based on expected chunk count per document
        )
        
        if not records:
            raise HTTPException(status_code=404, detail="Journal not found")
        
        # Sort chunks by chunk_index
        sorted_records = sorted(records, key=lambda x: x.payload["metadata"]["chunk_index"])
        
        # Extract metadata from first chunk (should be consistent across chunks)
        first_chunk = sorted_records[0].payload["metadata"]
        
        # Build metadata
        metadata = JournalMetadata(
            journal_id=journal_id,
            title=first_chunk.get("section_heading", "Unknown Title"),  # Use first section as title
            journal=first_chunk["journal"],
            publish_year=first_chunk["publish_year"],
            doi=first_chunk.get("doi"),
            link=first_chunk["link"],
            total_chunks=len(sorted_records)
        )
        
        # Build chunk responses
        chunks = []
        for record in sorted_records:
            chunk_metadata = record.payload["metadata"]
            chunks.append(JournalChunkResponse(
                id=chunk_metadata["id"],
                text=record.payload["page_content"],
                section=chunk_metadata["section_heading"],
                attributes=chunk_metadata["attributes"],
                citation_count=chunk_metadata.get("citation_count", 0)
            ))
        
        response = JournalResponse(
            metadata=metadata,
            chunks=chunks
        )
        
        logger.info(f"Retrieved journal {journal_id} with {len(chunks)} chunks")
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get journal {journal_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve journal: {str(e)}")

@app.post("/api/chat")
async def chat_with_llm(chat_request: ChatRequest):
    try:
        if not chat_request.query or not chat_request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
            
        # Create a SimilaritySearchRequest from the chat request
        search_request = SimilaritySearchRequest(
            query=chat_request.query,
            k=5,  # Default to 5 results, can be made configurable
            min_score=0.25  # Default minimum score
        )
        
        # Retrieve relevant context
        results = await similarity_search(search_request)
        
        if not results:
            return {"response": "No relevant information found in the knowledge base.", "sources": []}
        
        # Build context with citations
        context_parts = []
        sources = []
        for i, res in enumerate(results, 1):
            source_info = {
                "id": i, 
                "source": res.source, 
                "text": res.text,
                "section": res.section,
                "citation_count": res.citation_count
            }
            context_parts.append(f"[Source {i}: {source_info['source']}]\n{source_info['text']}")
            sources.append(source_info)
        
        context = "\n\n".join(context_parts)
        
        # Generate response with citations
        response = llm.invoke([
            ("system", "You are a research assistant. Based on the query you are given the following context from appropriate sources. Include inline citations like [1][2] when using information from sources. Answer straight to the point and be relevant. If the context doesn't contain relevant information, say so."),
            ("human", f"Context:\n{context}\n\nQuestion: {chat_request.query}")
        ])
        
        # Format the response
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        return {
            "response": response_text,
            "sources": sources,
            "query": chat_request.query
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred while processing your request")
    


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        collection_info = client.get_collection(collection_name)
        return {
            "status": "healthy",
            "vector_store": vector_store is not None,
            "collection": collection_name,
            "total_chunks": collection_info.points_count if collection_info else 0
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "vector_store": vector_store is not None
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)