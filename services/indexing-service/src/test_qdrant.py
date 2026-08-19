from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import uuid

client = QdrantClient("http://localhost:6333")
if not client.collection_exists("test_col"):
    client.create_collection("test_col", vectors_config={"dense": VectorParams(size=384, distance=Distance.COSINE)})

pt = PointStruct(
    id=str(uuid.uuid4()),
    vector={"dense": [0.1]*384},
    payload={"text": "hello"}
)
res = client.upsert("test_col", points=[pt])
print("Upsert result:", res)
print("Points count:", client.get_collection("test_col").points_count)
