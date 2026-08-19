import os
import sys
import uuid
import json
import requests
from io import BytesIO
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, SparseVector

sys.path.insert(0, "/home/arpan/Developments/Bragi/services/indexing-service/src")
from embed import embed_texts
import re
from collections import Counter

QDRANT_URL = "http://localhost:6333"
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
    print("Downloading English MSMARCO from HF...")
    url = "https://huggingface.co/datasets/ms_marco/resolve/main/data/train-00000-of-00003.parquet"
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    
    # We don't need to download the whole 1GB file. We can just read the first few MBs? No, read_parquet needs the whole file.
    # Let's use HF API with chunking or just download it.
    # Actually, we can use a smaller split! Let's use validation split which is smaller.
    url = "https://huggingface.co/datasets/ms_marco/resolve/main/data/validation-00000-of-00001.parquet"
    resp = requests.get(url)
    resp.raise_for_status()
    
    df = pd.read_parquet(BytesIO(resp.content))
    print(f"Loaded {len(df)} rows.")
    
    df = df.head(300)
    
    client = QdrantClient(url=QDRANT_URL)
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
        
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={"dense": VectorParams(size=384, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": {}}
    )
    
    batch_points = []
    queries_to_save = []
    processed = 0
    
    for idx, row in df.iterrows():
        q_text = row.get("query", "")
        if not q_text:
            continue
            
        passages = row.get("passages", {})
        texts = passages.get("passage_text", [])
        is_selected = passages.get("is_selected", [])
        
        if not texts:
            continue
            
        # Add to queries
        queries_to_save.append(q_text)
        
        for i, p_text in enumerate(texts):
            # Only ingest the selected passage if available, or all of them
            dense_vec = embed_texts([p_text])[0]
            indices, values = query_sparse_vector(p_text)
            sparse_vec = SparseVector(indices=indices, values=values) if indices else None
            
            vectors = {"dense": dense_vec}
            if sparse_vec:
                vectors["sparse"] = sparse_vec
                
            batch_points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=vectors,
                payload={
                    "text": p_text,
                    "chunk_strategy": "semantic",
                    "chunk_index": i,
                    "source_doc": str(idx),
                    "length": len(p_text)
                }
            ))
            processed += 1
            
            if len(batch_points) >= 50:
                client.upsert(collection_name=COLLECTION, points=batch_points)
                batch_points = []
                
        if len(queries_to_save) >= 200:
            break
            
    if batch_points:
        client.upsert(collection_name=COLLECTION, points=batch_points)
        
    print(f"Ingested {processed} passages.")
    
    with open("/home/arpan/Developments/Bragi/services/orchestrator/src/query_set.jsonl", "w") as f:
        for q in queries_to_save:
            f.write(json.dumps({"query": q, "category": "answerable"}) + "\n")
            
    print(f"Updated query_set.jsonl with {len(queries_to_save)} real English MSMARCO queries!")

if __name__ == "__main__":
    main()
