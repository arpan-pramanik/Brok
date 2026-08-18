import uuid
import pandas as pd
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams, HnswConfigDiff

QDRANT_URL = "http://localhost:6333"
COLLECTION = "msmarco_xi"
BATCH_SIZE = 256

def main():
    print(f"Connecting to Qdrant at {QDRANT_URL}...")
    client = QdrantClient(url=QDRANT_URL)

    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={"dense": VectorParams(size=384, distance=Distance.COSINE)},
        hnsw_config=HnswConfigDiff(m=16, ef_construct=64, on_disk=False),
        on_disk_payload=False
    )

    print("Loading FastEmbed BGESmallENV15 model...")
    embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    target = "/tmp/hinval.parquet"
    print(f"Loading dataset from {target}...")
    df = pd.read_parquet(target)
    print(f"Loaded {len(df)} records!")

    points = []
    seen_passages = set()

    for _, row in df.iterrows():
        query_id = row.get("query_id", "")
        passages = row.get("passages", {})
        
        texts = []
        if isinstance(passages, dict):
            texts = passages.get("English_passages", [])
            if len(texts) == 0:
                texts = passages.get("passage_text", [])

        for p_idx, text_raw in enumerate(texts):
            if not text_raw or not isinstance(text_raw, str):
                continue
            text_clean = text_raw.strip()
            if len(text_clean) < 15 or text_clean in seen_passages:
                continue

            seen_passages.add(text_clean)
            embeddings = list(embed_model.embed([text_clean]))
            emb = embeddings[0]

            point = PointStruct(
                id=str(uuid.uuid4()),
                vector={"dense": emb.tolist()},
                payload={
                    "text": text_clean,
                    "source_doc": f"msmarco_doc_{query_id}",
                    "chunk_index": p_idx,
                    "chunk_type": "semantic_passage",
                    "word_count": len(text_clean.split())
                }
            )
            points.append(point)

            if len(points) >= BATCH_SIZE:
                client.upsert(collection_name=COLLECTION, points=points)
                print(f"✅ Upserted batch into Qdrant (Total Indexed Passages: {len(seen_passages)})")
                points = []

            if len(seen_passages) >= 5000:
                break
        if len(seen_passages) >= 5000:
            break

    if points:
        client.upsert(collection_name=COLLECTION, points=points)

    print(f"✨ Ingestion Complete! Indexed {len(seen_passages)} passages into Qdrant HNSW index.")

if __name__ == "__main__":
    main()
