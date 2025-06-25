from fastapi import APIRouter
from .routes import upload, search, chat, health

# Create the main API router
router = APIRouter()

# Include all route modules without prefixes
router.include_router(upload.router, prefix="/api/upload", tags=["upload"])
router.include_router(search.router, prefix="/api", tags=["search"])
router.include_router(chat.router, prefix="/api/chat", tags=["chat"])
router.include_router(health.router, prefix="/api", tags=["health"])
