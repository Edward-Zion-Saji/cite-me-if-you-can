from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Request
from typing import Optional
import logging

from ...models.schemas import UploadRequest
from ...dependencies import get_upload_service, get_vector_store
from ...services.upload_service import UploadService
from ...core.vector_store import VectorStoreService

logger = logging.getLogger(__name__)

router = APIRouter()

@router.put("", status_code=202, tags=["upload"])
async def upload_journal(
    request: Request,
    file: Optional[UploadFile] = File(None),
    upload_service: UploadService = Depends(get_upload_service),
    vector_store: VectorStoreService = Depends(get_vector_store)
):
    """
    Upload journal chunks and generate embeddings.
    
    This endpoint accepts either:
    1. A file upload with a JSON array of chunks
    2. A form with 'file_url' pointing to a JSON file containing chunks
    
    The request should include a 'schema_version' field (default: "1.0")
    """
    try:
        # Parse form data
        form_data = await request.form()
        file_url = form_data.get('file_url')
        schema_version = form_data.get('schema_version', '1.0')
        
        # Get chunks from either file or URL
        chunks_data = []
        if file_url:
            chunks_data = await upload_service.fetch_chunks_from_url(file_url)
            logger.info(f"Fetched {len(chunks_data)} chunks from URL: {file_url}")
        elif file:
            chunks_data = await upload_service.process_uploaded_file(file)
            logger.info(f"Processed {len(chunks_data)} chunks from uploaded file")
        else:
            raise HTTPException(
                status_code=400,
                detail="Either 'file' (multipart/form-data) or 'file_url' (form field) must be provided"
            )
        
        if not chunks_data:
            raise HTTPException(status_code=400, detail="No valid chunks found to process")
        
        # Convert to documents and get journal citation tracking
        documents, journal_citations = upload_service.chunks_to_documents(chunks_data)
        
        # Store journal citation counts in a persistent store (e.g., database)
        # For now, we'll just log them
        logger.info(f"Initialized citation counts for {len(journal_citations)} journals")
        
        # Add documents to vector store (in a real app, this might be done asynchronously)
        try:
            vector_store.add_documents(documents)
            logger.info(f"Successfully indexed {len(documents)} chunks from {len(journal_citations)} journals")
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
        
        # Return 202 Accepted response
        return {
            "status": "accepted",
            "message": "Request accepted for processing",
            "chunk_count": len(documents),
            "journal_count": len(journal_citations),
            "schema_version": schema_version
        }
    
    except HTTPException as he:
        logger.warning(f"Client error in upload_journal: {str(he.detail)}")
        raise he
    except Exception as e:
        logger.error(f"Unexpected error in upload_journal: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )
