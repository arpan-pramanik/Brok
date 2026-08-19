import os
import sys
import uuid
import re
from collections import Counter
from pathlib import Path
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, SparseVector

sys.path.insert(0, str(Path(__file__).parent))
from embed import embed_texts

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = "msmarco_xi"

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

def main():
    print(f"Connecting to Qdrant at {QDRANT_URL}...")
    client = QdrantClient(url=QDRANT_URL)

    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
        
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={"dense": VectorParams(size=384, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": {}}
    )

    print("Loading local asmval.parquet...")
    df = pd.read_parquet("../indexing-service/asmval.parquet")
    print(f"Loaded {len(df)} rows.")
    
    batch_points = []
    processed = 0
    queries_extracted = []
    
    for idx, row in df.iterrows():
        eng_q = row.get("Eng_Query", "")
        if not eng_q or not isinstance(eng_q, str):
            continue
            
        passages = row.get("passages", {})
        if not isinstance(passages, dict):
            continue
            
        eng_passages = passages.get("English_passages", [])
        if len(eng_passages) == 0:
            continue
            
        queries_extracted.append(eng_q)
        print(f"Processing query {idx}: {eng_q[:30]}...")
        
        for i, passage in enumerate(eng_passages):
            passage_str = str(passage)
            if not passage_str:
                continue
                
            dense_vec = embed_texts([passage_str])[0]
            indices, values = query_sparse_vector(passage_str)
            sparse_vec = SparseVector(indices=indices, values=values) if indices else None
            
            vectors = {"dense": dense_vec}
            if sparse_vec:
                vectors["sparse"] = sparse_vec
                
            batch_points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=vectors,
                payload={
                    "text": passage_str,
                    "chunk_strategy": "semantic",
                    "chunk_index": i,
                    "source_doc": f"{idx}_{i}",
                    "length": len(passage_str)
                }
            ))
            processed += 1
            
            if len(batch_points) >= 100:
                client.upsert(collection_name=COLLECTION, points=batch_points)
                batch_points = []
                
        if len(queries_extracted) >= 200:
            break
            
    if batch_points:
        client.upsert(collection_name=COLLECTION, points=batch_points)
        
    print(f"Ingestion complete! Indexed {processed} passages.")
    
    with open("/home/arpan/Developments/Bragi/services/orchestrator/src/query_set.jsonl", "w") as f:
        import json
        for q in queries_extracted:
            f.write(json.dumps({"query": q, "category": "answerable"}) + "\n")
    print("Updated query_set.jsonl with real MSMARCO English queries!")

if __name__ == "__main__":
    main()
