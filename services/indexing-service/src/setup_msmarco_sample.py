import os
import sys
import uuid
import json
import urllib.request
from pathlib import Path
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
    print("Downloading MSMARCO sample queries...")
    queries_url = "https://raw.githubusercontent.com/microsoft/MSMARCO-Question-Answering/master/data/sample_queries.tsv"
    queries_data = urllib.request.urlopen(queries_url).read().decode('utf-8')
    
    print("Downloading MSMARCO sample passages...")
    passages_url = "https://raw.githubusercontent.com/microsoft/MSMARCO-Question-Answering/master/data/sample_passages.tsv"
    passages_data = urllib.request.urlopen(passages_url).read().decode('utf-8')
    
    # Process queries
    queries = {}
    for line in queries_data.strip().split("\n"):
        parts = line.split("\t")
        if len(parts) >= 2:
            queries[parts[0]] = parts[1]
            
    # Process passages
    passages = {}
    for line in passages_data.strip().split("\n"):
        parts = line.split("\t")
        if len(parts) >= 3:
            q_id = parts[0]
            passage_text = parts[2]
            if q_id not in passages:
                passages[q_id] = []
            passages[q_id].append(passage_text)

    # Ingest into Qdrant
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
    count = 0
    
    for q_id, q_text in queries.items():
        if count >= 200:
            break
            
        if q_id in passages:
            queries_to_save.append(q_text)
            for i, p_text in enumerate(passages[q_id]):
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
                        "source_doc": q_id,
                        "length": len(p_text)
                    }
                ))
            count += 1
            
            if len(batch_points) >= 50:
                client.upsert(collection_name=COLLECTION, points=batch_points)
                batch_points = []
                
    if batch_points:
        client.upsert(collection_name=COLLECTION, points=batch_points)
        
    print(f"Ingested passages for {count} queries.")
    
    # Write to query_set.jsonl
    with open("/home/arpan/Developments/Bragi/services/orchestrator/src/query_set.jsonl", "w") as f:
        for q in queries_to_save:
            f.write(json.dumps({"query": q, "category": "answerable"}) + "\n")
            
    print(f"Updated query_set.jsonl with {len(queries_to_save)} real MSMARCO queries!")

if __name__ == "__main__":
    main()
