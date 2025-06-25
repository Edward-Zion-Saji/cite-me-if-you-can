from fastapi import APIRouter
from .routes import upload, search, chat, health, citations


api_router = APIRouter()


api_router.include_router(upload.router, prefix="/upload", tags=["upload"])
api_router.include_router(search.router, tags=["search"])
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
api_router.include_router(health.router, tags=["health"])
api_router.include_router(citations.router, tags=["citations"])
