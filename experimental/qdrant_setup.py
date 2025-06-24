import json
from langchain_core.documents import Document
from qdrant_client import models, QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import FastEmbedEmbeddings

with open("sample_data.json", "r") as f:
    chunks = json.load(f)

documents = [
    Document(
        page_content=chunk["text"],
        metadata={
            "id": chunk["id"],
            "source": chunk["link"],
            "section": chunk["section_heading"],
            "attributes": chunk["attributes"],
            "year": chunk["publish_year"]
        }
    )
    for chunk in chunks
]

client = QdrantClient(url="http://localhost:6333", prefer_grpc=True)
collection_name = "data"


client.create_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(
        size=1024,
        distance=models.Distance.COSINE,
        on_disk=True
    ),
    quantization_config=models.BinaryQuantization(
        binary=models.BinaryQuantizationConfig(always_ram=True)
    )
)

embeddings = FastEmbedEmbeddings(model_name="thenlper/gte-large")
vector_store = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=embeddings,
    metadata_payload_key="metadata"  
)

vector_store.add_documents(documents)

collection_info = client.get_collection(collection_name)
print(f"Indexed {collection_info.points_count} documents")
