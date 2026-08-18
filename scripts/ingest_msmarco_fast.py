import os
import sys
import uuid
import gc
from datasets import load_dataset
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = "msmarco_xi"

def main():
    print(f"Connecting to Qdrant at {QDRANT_URL}...")
    client = QdrantClient(url=QDRANT_URL)

    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

    print("Loading FastEmbed BGESmallENV15 model...")
    embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    print("Loading mteb/msmarco official passage corpus...")
    ds = load_dataset("mteb/msmarco", "corpus", split="corpus", streaming=True)

    print("Streaming real MS MARCO passages directly into Qdrant...")
    batch_texts = []
    batch_metas = []
    total_indexed = 0
    TARGET_PASSAGES = 200000

    for idx, item in enumerate(ds):
        if total_indexed >= TARGET_PASSAGES:
            break

        p_str = item.get("text", "").strip()
        doc_id = item.get("_id", idx)

        if not p_str or len(p_str) < 20:
            continue

        batch_texts.append(p_str)
        batch_metas.append({
            "text": p_str,
            "source_doc": f"msmarco_doc_{doc_id}",
            "chunk_index": 0
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
            if total_indexed % 1000 == 0:
                print(f"✅ Indexed {total_indexed}/{TARGET_PASSAGES} passages into Qdrant...")
            batch_texts = []
            batch_metas = []
            gc.collect()

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

    print(f"🎉 Fully indexed {total_indexed} real MS MARCO passages into Qdrant collection '{COLLECTION}'!")

if __name__ == "__main__":
    main()
