import os
import sys
import hashlib
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    SparseVectorParams,
    SparseVector,
)
from collections import Counter
import math
import re

sys.path.insert(0, str(Path(__file__).parent))
from embed import embed_texts
from chunkers.fixed_overlap import fixed_overlap_chunk
from chunkers.semantic import semantic_chunk
from chunkers.structural import structural_chunk

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("COLLECTION_NAME", "bragi_goa")
CORPUS_DIR = os.getenv("CORPUS_DIR", str(Path(__file__).parent.parent / "data" / "corpus"))
DENSE_DIM = 384


def build_sparse_vector(text: str, idf: dict[str, float]) -> tuple[list[int], list[float]]:
    tokens = re.findall(r"\w+", text.lower())
    tf = Counter(tokens)
    doc_len = len(tokens)
    if doc_len == 0:
        return [], []
    indices = []
    values = []
    for token, count in tf.items():
        if token in idf:
            token_id = hash(token) % (2**31)
            tf_score = count / doc_len
            indices.append(token_id)
            values.append(tf_score * idf[token])
    return indices, values


def compute_idf(all_chunks: list[str]) -> dict[str, float]:
    n = len(all_chunks)
    doc_freq: dict[str, int] = Counter()
    for chunk in all_chunks:
        tokens = set(re.findall(r"\w+", chunk.lower()))
        for token in tokens:
            doc_freq[token] += 1
    return {token: math.log(n / (1 + df)) for token, df in doc_freq.items()}


def load_corpus(corpus_dir: str) -> list[tuple[str, str]]:
    docs = []
    for f in sorted(Path(corpus_dir).glob("*.md")):
        docs.append((f.stem, f.read_text()))
    return docs


def chunk_document(doc_name: str, text: str) -> list[dict]:
    chunks = []
    for strategy, chunker in [
        ("fixed_overlap", fixed_overlap_chunk),
        ("semantic", semantic_chunk),
        ("structural", structural_chunk),
    ]:
        for i, chunk_text in enumerate(chunker(text)):
            chunk_id = hashlib.md5(f"{doc_name}:{strategy}:{i}".encode()).hexdigest()
            chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "source_doc": doc_name,
                "strategy": strategy,
                "chunk_index": i,
            })
    return chunks


def build_index():
    print(f"loading corpus from {CORPUS_DIR}")
    docs = load_corpus(CORPUS_DIR)
    print(f"loaded {len(docs)} documents")

    all_chunks = []
    for doc_name, text in docs:
        all_chunks.extend(chunk_document(doc_name, text))
    print(f"created {len(all_chunks)} chunks across all strategies")

    texts = [c["text"] for c in all_chunks]
    print("computing IDF...")
    idf = compute_idf(texts)

    print("generating embeddings...")
    embeddings = embed_texts(texts)

    print("connecting to qdrant...")
    client = QdrantClient(path=str(Path(__file__).parent.parent.parent.parent.parent / "qdrant_data"))

    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={"dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams()},
    )

    points = []
    for i, chunk in enumerate(all_chunks):
        sparse_indices, sparse_values = build_sparse_vector(chunk["text"], idf)
        points.append(
            PointStruct(
                id=i,
                vector={
                    "dense": embeddings[i],
                    "sparse": SparseVector(indices=sparse_indices, values=sparse_values),
                },
                payload={
                    "chunk_id": chunk["id"],
                    "text": chunk["text"],
                    "source_doc": chunk["source_doc"],
                    "strategy": chunk["strategy"],
                    "chunk_index": chunk["chunk_index"],
                },
            )
        )

    batch_size = 100
    for i in range(0, len(points), batch_size):
        client.upsert(collection_name=COLLECTION, points=points[i : i + batch_size])
    print(f"indexed {len(points)} chunks into qdrant collection '{COLLECTION}'")


if __name__ == "__main__":
    build_index()
