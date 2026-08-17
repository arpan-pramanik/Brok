import os
import sys
import re
import math
from pathlib import Path
from collections import Counter
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "indexing-service" / "src"))
from embed import embed_query

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("COLLECTION_NAME", "msmarco_xi")


_client = None
def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(path=str(Path(__file__).parent.parent.parent.parent / "qdrant_data"))
    return _client


def query_sparse_vector(query: str) -> tuple[list[int], list[float]]:
    tokens = re.findall(r"\w+", query.lower())
    tf = Counter(tokens)
    doc_len = len(tokens)
    if doc_len == 0:
        return [], []
    indices = []
    values = []
    for token, count in tf.items():
        token_id = hash(token) % (2**31)
        indices.append(token_id)
        values.append(count / doc_len)
    return indices, values


def dense_search(client: QdrantClient, query: str, limit: int = 20) -> list[dict]:
    vector = embed_query(query)
    results = client.query_points(
        collection_name=COLLECTION,
        query=vector,
        using="dense",
        limit=limit,
        with_payload=True,
    )
    return [
        {"id": r.id, "score": r.score, **r.payload}
        for r in results.points
    ]


def sparse_search(client: QdrantClient, query: str, limit: int = 20) -> list[dict]:
    indices, values = query_sparse_vector(query)
    if not indices:
        return []
    results = client.query_points(
        collection_name=COLLECTION,
        query=SparseVector(indices=indices, values=values),
        using="sparse",
        limit=limit,
        with_payload=True,
    )
    return [
        {"id": r.id, "score": r.score, **r.payload}
        for r in results.points
    ]


def hybrid_search(query: str, limit: int = 20) -> tuple[list[dict], list[dict]]:
    client = get_client()
    dense_results = dense_search(client, query, limit)
    sparse_results = sparse_search(client, query, limit)
    return dense_results, sparse_results
