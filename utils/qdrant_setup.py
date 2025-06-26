import os
import sys
from qdrant_client import models, QdrantClient


QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
collection_name = "research_data"

print(f"Connecting to Qdrant at {QDRANT_URL}")
try:
    client = QdrantClient(url=QDRANT_URL, prefer_grpc=True)
    client.get_collections()
except Exception as e:
    print(f"Error connecting to Qdrant at {QDRANT_URL}: {str(e)}")
    sys.exit(1)

# Check if collection already exists
collections = client.get_collections()
collection_names = [collection.name for collection in collections.collections]

if collection_name in collection_names:
    print(f"Collection '{collection_name}' already exists. Skipping creation.")
else:
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