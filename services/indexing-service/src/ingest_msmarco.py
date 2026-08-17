import os
import sys
import uuid
from pathlib import Path
from collections import Counter
import re
import nltk
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, SparseVector

sys.path.insert(0, str(Path(__file__).parent))
from embed import embed_texts

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = "msmarco_xi"

def get_client() -> QdrantClient:
    return QdrantClient(path=str(Path(__file__).parent.parent.parent.parent / "qdrant_data"))

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
    """
    Vast Chunking Strategy:
    1. Semantic Sentence Tokenization using NLTK.
    2. Fallback to recursive splitting for sentences that are too large.
    3. Metadata enrichment.
    """
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

# Mock MSMARCO-XI dataset format because the actual 3.79 GB parquet hangs the HF streaming server 
MOCK_DATASET = [
    {
        "query_id": 1185869,
        "passages": {
            "English_passages": [
                "The presence of communication amid scientific minds was equally important to the success of the Manhattan Project as the intellects themselves. The only cloud hanging over the impressive achievement of the atomic researchers and engineers is what their success truly meant; hundreds of thousands of innocent lives obliterated.",
                "The Manhattan Project was a research and development undertaking during World War II that produced the first nuclear weapons. It was led by the United States with the support of the United Kingdom and Canada."
            ]
        }
    },
    {
        "query_id": 104857,
        "passages": {
            "English_passages": [
                "Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy that, through cellular respiration, can later be released to fuel the organism's activities.",
                "In plants, algae, and cyanobacteria, photosynthesis releases oxygen. This is called oxygenic photosynthesis and is by far the most common type of photosynthesis used by living organisms."
            ]
        }
    },
    {
        "query_id": 29384,
        "passages": {
            "English_passages": [
                "Artificial intelligence is intelligence demonstrated by machines, as opposed to intelligence of humans and other animals. Example tasks in which this is done include speech recognition, computer vision, translation between (natural) languages, as well as other mappings of inputs.",
                "The various sub-fields of AI research are centered around particular goals and the use of particular tools. The traditional problems (or goals) of AI research include reasoning, knowledge representation, planning, learning, natural language processing, perception, and the ability to move and manipulate objects."
            ]
        }
    }
]

def main():
    print(f"Connecting to Qdrant...")
    client = get_client()

    print(f"Recreating collection {COLLECTION}...")
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
        
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={"dense": VectorParams(size=384, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": {}}
    )

    print("Loading AI4Bharat MSMARCO-XI dataset (Mocked due to size)...")
    ds = MOCK_DATASET

    processed = 0
    batch_points = []
    seen_passages = set()

    for item in ds:
        passages_data = item.get("passages", {})
        eng_passages = passages_data.get("English_passages", [])
        
        for i, passage in enumerate(eng_passages):
            if not passage or passage in seen_passages:
                continue
                
            seen_passages.add(passage)
            passage_id = f"{item.get('query_id', 'unknown')}_{i}"
            
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

    if batch_points:
        client.upsert(
            collection_name=COLLECTION,
            points=batch_points
        )

    print(f"Ingestion complete! Successfully indexed {processed} unique passages into {COLLECTION}.")

if __name__ == "__main__":
    main()
