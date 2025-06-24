from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from pydantic import BaseModel
import json
import uuid
from langchain_core.documents import Document
from qdrant_client import QdrantClient, models
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_sambanova import ChatSambaNovaCloud
import os

app = FastAPI()

client = QdrantClient(url="http://localhost:6333", prefer_grpc=True)
collection_name = "data"
embeddings = FastEmbedEmbeddings(model_name="thenlper/gte-large")
vector_store = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=embeddings,
    metadata_payload_key="metadata"
)

os.environ['SAMBANOVA_API_KEY'] = os.getenv('SAMBANOVA_API_KEY')
llm = ChatSambaNovaCloud(model="DeepSeek-R1")

class UploadRequest(BaseModel):
    chunks: list
    schema_version: str

class SearchRequest(BaseModel):
    query: str
    k: int = 10
    min_score: float = 0.25

@app.put("/api/upload")
async def upload_journal(request: UploadRequest):
    documents = [
        Document(
            page_content=chunk["text"],
            metadata={
                "id": chunk["id"],
                "source": chunk["link"],
                "section": chunk["section_heading"],
                "attributes": chunk["attributes"],
                "year": chunk["publish_year"],
                "source_doc_id": chunk["source_doc_id"]
            }
        ) for chunk in request.chunks
    ]
    
    vector_store.add_documents(documents)
    return {"status": "accepted", "message": f"{len(documents)} chunks indexed"}

@app.post("/api/similarity_search")
async def semantic_search(request: SearchRequest):
    """Perform semantic similarity search"""
    query_embedding = embeddings.embed_query(request.query)
    results = client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=request.k,
        score_threshold=request.min_score,
        with_payload=True
    )
    
    return [{
        "id": result.payload["metadata"]["id"],
        "score": result.score,
        "text": result.payload["page_content"],
        "source": result.payload["metadata"]["source"],
        "section": result.payload["metadata"]["section"]
    } for result in results]

@app.get("/api/{journal_id}")
async def get_journal(journal_id: str):
    results = client.scroll(
        collection_name=collection_name,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(
                key="metadata.source_doc_id",
                match=models.MatchValue(value=journal_id))
            ]
        ),
        with_payload=True,
        limit=100
    )
    
    if not results:
        raise HTTPException(status_code=404, detail="Journal not found")
    
    return [{
        "id": record.payload["metadata"]["id"],
        "text": record.payload["metadata"]["text"],
        "section": record.payload["metadata"]["section"],
        "attributes": record.payload["metadata"]["attributes"]
    } for record in results[0]]


class ChatRequest(BaseModel):
    query: str

@app.post("/api/chat")
async def chat_with_llm(request: ChatRequest):

    results = await semantic_search(SearchRequest(query=request.query))
    context = "\n".join([f"[Source: {res['source']}]\n{res['text']}" for res in results])
    
    response = llm.invoke([
        ("system", "Answer using ONLY the context below:"),
        ("human", f"Context:\n{context}\n\nQuestion: {request.query}")
    ])
    
    return {"response": response.content}
