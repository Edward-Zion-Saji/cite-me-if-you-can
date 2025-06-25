import json
from langchain_core.documents import Document
from qdrant_client import models, QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import FastEmbedEmbeddings

# Load the pre-chunked data
with open("sample_data.json", "r") as f:
    chunks = json.load(f)

# Convert chunks to LangChain Document objects
documents = [
    Document(
        page_content=chunk["text"],
        metadata={
            "id": chunk["id"],
            "source_doc_id": chunk["source_doc_id"],
            "chunk_index": chunk["chunk_index"],
            "section_heading": chunk["section_heading"],
            "journal": chunk["journal"],
            "publish_year": chunk["publish_year"],
            "usage_count": chunk["usage_count"],
            "attributes": chunk["attributes"],
            "link": chunk["link"],
            # Add DOI if it exists (only for some documents)
            **({"doi": chunk["doi"]} if "doi" in chunk else {})
        }
    )
    for chunk in chunks
]

# Initialize Qdrant client
client = QdrantClient(url="http://localhost:6333", prefer_grpc=True)
collection_name = "research_data"

# Check if collection already exists
collections = client.get_collections()
collection_names = [collection.name for collection in collections.collections]

if collection_name in collection_names:
    print(f"Collection '{collection_name}' already exists. Skipping creation.")
    # If you want to delete and recreate, uncomment the following:
    # client.delete_collection(collection_name)
    # print(f"Deleted existing collection '{collection_name}'")
else:
    # Create collection with optimized settings
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=1024,  # gte-large embedding dimension
                distance=models.Distance.COSINE,
                on_disk=True
            ),
            quantization_config=models.BinaryQuantization(
                binary=models.BinaryQuantizationConfig(always_ram=True)
            )
        )
        print(f"Created collection '{collection_name}'")
    except Exception as e:
        print(f"Error creating collection: {e}")
        print("Trying to continue with existing collection...")

""" # Initialize embeddings model
embeddings = FastEmbedEmbeddings(model_name="thenlper/gte-large")


vector_store = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=embeddings,
    metadata_payload_key="metadata"
)

print("Adding documents to vector store...")
vector_store.add_documents(documents)

collection_info = client.get_collection(collection_name)
print(f"Successfully indexed {collection_info.points_count} documents")


print("\nSample document metadata:")
for i, doc in enumerate(documents[:2]):
    print(f"Document {i+1}:")
    print(f"  ID: {doc.metadata['id']}")
    print(f"  Source: {doc.metadata['source_doc_id']}")
    print(f"  Section: {doc.metadata['section_heading']}")
    print(f"  Attributes: {doc.metadata['attributes']}")
    print(f"  Text preview: {doc.page_content[:100]}...")
    print()


 """