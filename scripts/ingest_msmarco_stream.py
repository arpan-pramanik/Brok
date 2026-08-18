import os
import sys
import uuid
import gc
import json
import urllib.request
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = "msmarco_xi"
PARQUET_API_URL = "https://huggingface.co/api/datasets/ai4bharat/MSMARCO-XI/parquet"

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

    print("Fetching parquet manifest from HuggingFace...")
    try:
        req = urllib.request.Request(PARQUET_API_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp:
            manifest = json.loads(resp.read().decode())
        urls = manifest.get("default", {}).get("train", [])
    except Exception as e:
        print(f"Error fetching manifest: {e}")
        urls = []

    print(f"Found {len(urls)} parquet files for MS MARCO dataset.")

    total_indexed = 0
    TARGET_PASSAGES = 50000

    for url_idx, file_url in enumerate(urls):
        if total_indexed >= TARGET_PASSAGES:
            break
        
        print(f"Downloading parquet file [{url_idx+1}/{len(urls)}] to /tmp/chunk.parquet...")
        tmp_path = "/tmp/chunk.parquet"
        try:
            req_file = urllib.request.Request(file_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req_file) as resp, open(tmp_path, "wb") as out_file:
                import shutil
                shutil.copyfileobj(resp, out_file, length=1024*1024)

            print(f"Processing row-batches from /tmp/chunk.parquet...")
            pf = pq.ParquetFile(tmp_path)
            
            for row_group in range(pf.num_row_groups):
                if total_indexed >= TARGET_PASSAGES:
                    break

                table = pf.read_row_group(row_group)
                p_dict = table.to_pydict()
                passages_col = p_dict.get("passages", [])
                query_ids = p_dict.get("query_id", [])

                batch_texts = []
                batch_metas = []

                for i, passages in enumerate(passages_col):
                    passage_texts = []
                    if isinstance(passages, dict):
                        passage_texts = passages.get("passage_text", [])
                    elif isinstance(passages, list):
                        passage_texts = [p.get("passage_text", "") if isinstance(p, dict) else str(p) for p in passages]

                    q_id = query_ids[i] if i < len(query_ids) else i

                    for p_text in passage_texts:
                        p_str = str(p_text).strip()
                        if not p_str or len(p_str) < 20:
                            continue

                        batch_texts.append(p_str)
                        batch_metas.append({
                            "text": p_str,
                            "source_doc": f"msmarco_doc_{q_id}",
                            "chunk_index": 0
                        })

                        if len(batch_texts) >= 100:
                            embeddings = list(embed_model.embed(batch_texts))
                            points = [
                                PointStruct(
                                    id=str(uuid.uuid4()),
                                    vector=emb.tolist(),
                                    payload=meta
                                )
                                for emb, meta in zip(embeddings, batch_metas)
                            ]
                            client.upsert(collection_name=COLLECTION, points=points)
                            total_indexed += len(points)
                            print(f"Indexed {total_indexed}/{TARGET_PASSAGES} passages into Qdrant...")
                            batch_texts = []
                            batch_metas = []

                del table
                del p_dict
                gc.collect()

            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        except Exception as e:
            print(f"Error processing batch: {e}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            continue

    print(f"🎉 Fully indexed {total_indexed} MS MARCO passages into Qdrant!")

if __name__ == "__main__":
    main()
