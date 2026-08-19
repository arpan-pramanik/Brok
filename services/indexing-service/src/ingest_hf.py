import os
import sys
import uuid
import re
from collections import Counter
from pathlib import Path
import nltk
import pandas as pd
from datasets import load_dataset
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, SparseVector

sys.path.insert(0, str(Path(__file__).parent))
from embed import embed_texts

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = "msmarco_xi"

def get_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)

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

def recursive_chunk(text: str, max_len: int = 512, overlap: int = 64) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_len
        chunks.append(text[start:end])
        start += max_len - overlap
    return chunks

def multi_strategy_chunk(passage: str, source_id: str) -> list[dict]:
    chunks = []
    try:
        sentences = nltk.sent_tokenize(passage)
    except Exception:
        sentences = [passage]

    buffer = ""
    chunk_index = 0
    MAX_CHUNK_CHARS = 256

    for sentence in sentences:
        if len(buffer) + len(sentence) <= MAX_CHUNK_CHARS:
            buffer += " " + sentence if buffer else sentence
        else:
            if buffer:
                chunks.append({
                    "text": buffer.strip(),
                    "chunk_strategy": "semantic",
                    "chunk_index": chunk_index,
                    "source_id": source_id,
                    "length": len(buffer.strip())
                })
                chunk_index += 1
                buffer = ""
            
            if len(sentence) > MAX_CHUNK_CHARS:
                sub_chunks = recursive_chunk(sentence, MAX_CHUNK_CHARS, overlap=32)
                for sub in sub_chunks:
                    chunks.append({
                        "text": sub.strip(),
                        "chunk_strategy": "recursive_fixed",
                        "chunk_index": chunk_index,
                        "source_id": source_id,
                        "length": len(sub.strip())
                    })
                    chunk_index += 1
            else:
                buffer = sentence
    
    if buffer:
        chunks.append({
            "text": buffer.strip(),
            "chunk_strategy": "semantic",
            "chunk_index": chunk_index,
            "source_id": source_id,
            "length": len(buffer.strip())
        })
        
    return chunks

def process_dataframe(df: pd.DataFrame, client: QdrantClient):
    processed = 0
    batch_points = []
    seen_passages = set()
    
    for idx, row in df.iterrows():
        # Using correct fields from original script logic
        passages_data = row.get("passages", [])
        eng_passages = []
        if isinstance(passages_data, dict):
             eng_passages = passages_data.get("English_passages", [])
        elif isinstance(passages_data, list):
             eng_passages = passages_data
        elif isinstance(passages_data, str):
             eng_passages = [passages_data]
             
        query_id = row.get("query_id", f"unknown_{idx}")

        for i, passage in enumerate(eng_passages):
            if isinstance(passage, dict):
                passage = passage.get("text", "")
            
            if not passage or not isinstance(passage, str) or passage in seen_passages:
                continue
                
            seen_passages.add(passage)
            passage_id = f"{query_id}_{i}"
            
            chunks = multi_strategy_chunk(passage, passage_id)
            
            for chunk_data in chunks:
                text = chunk_data["text"]
                if not text:
                    continue
                    
                dense_vec = embed_texts([text])[0]
                indices, values = query_sparse_vector(text)
                sparse_vec = SparseVector(indices=indices, values=values) if indices else None
                
                point_id = str(uuid.uuid4())
                
                vectors = {"dense": dense_vec}
                if sparse_vec:
                    vectors["sparse"] = sparse_vec
                    
                batch_points.append(PointStruct(
                    id=point_id,
                    vector=vectors,
                    payload={
                        "text": text,
                        "chunk_strategy": chunk_data["chunk_strategy"],
                        "chunk_index": chunk_data["chunk_index"],
                        "source_id": chunk_data["source_id"],
                        "length": chunk_data["length"],
                        "language": "en"
                    }
                ))
            
            processed += 1
            
            if len(batch_points) >= 100:
                client.upsert(collection_name=COLLECTION, points=batch_points)
                batch_points = []
                
    if batch_points:
        client.upsert(collection_name=COLLECTION, points=batch_points)
        
    return processed

def main():
    print(f"Connecting to Qdrant at {QDRANT_URL}...")
    client = get_client()

    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config={"dense": VectorParams(size=384, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": {}}
        )

    print("Loading AI4Bharat/MSMARCO-XI from HuggingFace...")
    # Some datasets don't have config name. Let's just try loading it.
    ds = load_dataset("AI4Bharat/MSMARCO-XI", "default", split="train", streaming=True)
    
    # We will just parse 1000 rows
    print("Converting to DataFrame...")
    items = []
    count = 0
    for item in ds:
        # MS MARCO v1.1 has 'passages' containing { 'is_selected': [], 'url': [], 'passage_text': [] }
        passages = item.get("passages", {})
        texts = passages.get("passage_text", [])
        
        items.append({
            "query_id": item.get("query_id", str(count)),
            "passages": texts
        })
        count += 1
        if count >= 1000:
            break
            
    df = pd.DataFrame(items)
    print(f"Loaded {len(df)} rows. Ingesting...")
    
    processed = process_dataframe(df, client)
    print(f"\nIngestion complete! Successfully indexed {processed} unique passages into {COLLECTION}.")

if __name__ == "__main__":
    main()
