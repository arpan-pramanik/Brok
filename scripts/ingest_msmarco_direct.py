import os
import sys
import uuid
from datasets import load_dataset
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = "msmarco_xi"

def main():
    print(f"Connecting to Qdrant at {QDRANT_URL}...")
    client = QdrantClient(url=QDRANT_URL)

    if client.collection_exists(COLLECTION):
        print(f"Collection {COLLECTION} already exists. Re-creating...")
        client.delete_collection(COLLECTION)

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={"dense": VectorParams(size=384, distance=Distance.COSINE)}
    )

    print("Loading FastEmbed BGESmallENV15 model...")
    embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    print("Loading ai4bharat/MSMARCO-XI dataset...")
    try:
        ds = load_dataset("ai4bharat/MSMARCO-XI", split="train", streaming=True)
    except Exception as e:
        print(f"Fallback loading ms_marco: {e}")
        ds = load_dataset("ms_marco", "v2.1", split="train", streaming=True)

    print("Ingesting passages into Qdrant...")
    batch_texts = []
    batch_metas = []
    total_indexed = 0
    TARGET_PASSAGES = 2000

    for idx, item in enumerate(ds):
        if total_indexed >= TARGET_PASSAGES:
            break
        
        passages = item.get("passages", {})
        if isinstance(passages, dict):
            passage_texts = passages.get("passage_text", [])
        elif isinstance(passages, list):
            passage_texts = [p.get("passage_text", "") if isinstance(p, dict) else str(p) for p in passages]
        else:
            continue

        for p_text in passage_texts:
            if not p_text or len(p_text.strip()) < 20:
                continue
            
            batch_texts.append(p_text.strip())
            batch_metas.append({
                "text": p_text.strip(),
                "source_doc": f"doc_{item.get('query_id', idx)}",
                "chunk_index": 0,
                "language": "en"
            })

            if len(batch_texts) >= 100:
                embeddings = list(embed_model.embed(batch_texts))
                points = [
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector={"dense": emb.tolist()},
                        payload=meta
                    )
                    for emb, meta in zip(embeddings, batch_metas)
                ]
                client.upsert(collection_name=COLLECTION, points=points)
                total_indexed += len(points)
                print(f"Indexed {total_indexed}/{TARGET_PASSAGES} passages into Qdrant...")
                batch_texts = []
                batch_metas = []

            if total_indexed >= TARGET_PASSAGES:
                break

    if batch_texts:
        embeddings = list(embed_model.embed(batch_texts))
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector={"dense": emb.tolist()},
                payload=meta
            )
            for emb, meta in zip(embeddings, batch_metas)
        ]
        client.upsert(collection_name=COLLECTION, points=points)
        total_indexed += len(points)

    print(f"Successfully indexed {total_indexed} passages into Qdrant collection '{COLLECTION}'.")

if __name__ == "__main__":
    main()
