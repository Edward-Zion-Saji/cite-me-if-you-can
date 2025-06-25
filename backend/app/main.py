import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from .config import settings
from .api.routes import upload, search, chat, health
from .dependencies import get_vector_store, get_embedding_service


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_application() -> FastAPI:
    app = FastAPI(
        title="Journal RAG API",
        version="1.0.0",
        debug=settings.DEBUG
    )
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], 
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize services
    @app.on_event("startup")
    async def startup_event():
        try:
            # Initialize vector store with embeddings
            embedding_service = get_embedding_service()
            vector_store = get_vector_store(embedding_service)
            logger.info("Services initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize services: {e}")
            raise
    
    # Include routes
    app.include_router(upload.router, prefix=settings.API_PREFIX)
    app.include_router(search.router, prefix=settings.API_PREFIX)
    app.include_router(chat.router, prefix=settings.API_PREFIX)
    app.include_router(health.router)  
    return app

app = create_application()


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
