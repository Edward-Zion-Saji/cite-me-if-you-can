# Cite Me If You Can

A research assistan tool that uses RAG.

## For Ingestion Pipeline

Please refer to the [Ingestion Pipeline](./ingestion-pipeline.md) doc for details on the ingestion pipeline.

## Table of Contents
- [Running the Application](#running)
- [Project Structure](#project-structure)
- [Backend API](#backend-api)
- [Frontend](#frontend)

## Running

### Prerequisites
- Docker 
- Python 3.10+

### Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/Edward-Zion-Saji/cite-me-if-you-can.git
   cd cite-me-if-you-can
   ```

2. Copy the backend/.env.example file to backend/.env and fill in the required API key for SAMBANOVA Deepseek R1 (https://sambanova.ai/):
   ```bash
   SAMBANOVA_API_KEY=XXXXXXXXXXXXXXXXX
   ```

3. Start the application using Docker Compose:
   ```bash
   docker compose up --build
   ```

4. Access the application:
   - Frontend: http://localhost:5001
   - Backend API: http://localhost:8000
   - Qdrant Dashboard: http://localhost:6333

5. Add sample pre-chunked papers to Qdrant:
   - Go to the Upload tab in the front end
   - Upload the [sample data file](./qdrant/sample_data.json)



## Project Structure

```
cite-me-if-you-can/
├── backend/           # FastAPI backend service
│   ├── app/           # Application code
│   ├── tests/         # Backend tests
│   └── README.md      # Backend documentation
├── frontend/          # Flask frontend application
│   ├── static/        # Static files
│   ├── templates/     # HTML templates
│   └── app.py         # Flask application
└── docker-compose.yml # Docker Compose configuration
```

## Backend API

For detailed API documentation, please refer to the [Backend README](./backend/README.md).

Key features:
- RESTful API endpoints for managing papers and citations
- Integration with Qdrant vector database for semantic search
- Authentication and rate limiting
- Comprehensive error handling

## Frontend

The frontend is a Flask application that provides a web interface for interacting with the backend API.

### Features
- Modern, responsive UI built with Tailwind CSS
- Real-time search and filtering
- Citation tracking and statistics
- Paper upload and management

