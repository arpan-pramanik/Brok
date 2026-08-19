import os
import sys
import uuid
import re
from collections import Counter
from pathlib import Path
import nltk
import pandas as pd
import requests
from io import BytesIO
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

    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config={"dense": VectorParams(size=384, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": {}}
        )

    # Download one small validation shard
    url = "https://huggingface.co/datasets/AI4Bharat/MSMARCO-XI/resolve/main/validation/asmval.parquet"
    print(f"Downloading {url} ...")
    resp = requests.get(url)
    resp.raise_for_status()
    
    df = pd.read_parquet(BytesIO(resp.content))
    print(f"Loaded {len(df)} rows.")
    
    # Take first 10 rows to debug
    df = df.head(10)
    
    batch_points = []
    processed = 0
    queries_extracted = []
    
    for idx, row in df.iterrows():
        eng_q = row.get("Eng_Query", "")
        if eng_q and isinstance(eng_q, str):
            queries_extracted.append(eng_q)
            
        passages_data = row.get("passages", {})
        eng_passages = []
        if isinstance(passages_data, dict):
             eng_passages = passages_data.get("passage_text", [])
        elif isinstance(passages_data, list):
             eng_passages = passages_data
        elif isinstance(passages_data, str):
             eng_passages = [passages_data]
             
        query_id = row.get("query_id", f"unknown_{idx}")

        for i, passage in enumerate(eng_passages):
            if isinstance(passage, dict):
                passage = passage.get("text", "")
            
            if not passage:
                continue
            passage = str(passage)
                
            dense_vec = embed_texts([passage])[0]
            indices, values = query_sparse_vector(passage)
            sparse_vec = SparseVector(indices=indices, values=values) if indices else None
            
            point_id = str(uuid.uuid4())
            vectors = {"dense": dense_vec}
            if sparse_vec:
                vectors["sparse"] = sparse_vec
                
            batch_points.append(PointStruct(
                id=point_id,
                vector=vectors,
                payload={
                    "text": passage,
                    "chunk_strategy": "semantic",
                    "chunk_index": 0,
                    "source_doc": f"{query_id}_{i}",
                    "length": len(passage)
                }
            ))
            processed += 1
            
            if len(batch_points) >= 10:
                res = client.upsert(collection_name=COLLECTION, points=batch_points)
                print("Upsert result:", res)
                batch_points = []
                
    if batch_points:
        res = client.upsert(collection_name=COLLECTION, points=batch_points)
        print("Final upsert result:", res)
        
    print(f"Ingestion complete! Indexed {processed} passages.")
    
    # Write some of the queries out to update the query_set.jsonl
    with open("/home/arpan/Developments/Bragi/services/orchestrator/src/query_set.jsonl", "w") as f:
        import json
        for q in list(set(queries_extracted))[:200]:
            f.write(json.dumps({"query": q, "category": "answerable"}) + "\n")
    print("Updated query_set.jsonl with real MSMARCO queries from this shard!")

if __name__ == "__main__":
    main()
