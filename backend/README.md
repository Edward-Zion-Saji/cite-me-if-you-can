# Backend API Documentation

This document provides detailed information about the backend API endpoints, request/response formats, and usage examples.

## Table of Contents
- [API Endpoints](#api-endpoints)
- [Request/Response Formats](#requestresponse-formats)
- [Error Handling](#error-handling)

## API Endpoints

Swagger UI: http://localhost:8000/docs
### Search

- `POST /api/similarity_search` - Semantic similarity search with vector embeddings
- `GET /api/search/{journal_id}` - Get metadata and chunks for a specific journal

### Citations
- `GET /api/citations/stats` - Get top cited documents
- `GET /api/citations/document/{doc_id}` - Get citation count for a specific document

### Upload
- `PUT /api/upload` - Upload journal chunks and generate embeddings
  - Accepts either file upload or file URL
  - Processes chunks asynchronously

### Chat
- `POST /api/chat` - Chat with the LLM using RAG (Retrieval-Augmented Generation)

### Health
- `GET /api/health` - Service health check
  - Returns vector store status and collection info



## Request/Response Formats

### 1. Similarity Search
```http
POST /api/similarity_search
Content-Type: application/json

{
  "query": "your search query",
  "k": 10,
  "min_score": 0.25
}
```

### 2. Upload Journal Chunks
```http
PUT /api/upload
Content-Type: multipart/form-data

# Either include a file:
file: [your_file.json]

# Or provide a file URL:
file_url: "https://example.com/chunks.json"

# Optional:
schema_version: "1.0"
```

### 3. Chat with LLM
```http
POST /api/chat
Content-Type: application/json

{
  "query": "Your question about the research"
}
```

### 4. Get Journal by ID
```http
GET /api/search/{journal_id}
```

### 5. Get Citation Statistics
```http
GET /api/citations/stats?limit=10
```

### 6. Get Document Citations
```http
GET /api/citations/document/{doc_id}
```







