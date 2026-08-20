import os
import sys
import uuid
import re
import argparse
import time
from collections import Counter
from datasets import load_dataset
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, SparseVector
from fastembed import TextEmbedding

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("COLLECTION_NAME", "msmarco_xi")

def parse_args():
    parser = argparse.ArgumentParser(description="Ingest AI4Bharat/MSMARCO-XI dataset into Qdrant")
    parser.add_argument("--split", type=str, default="validation", choices=["validation", "train"], help="Dataset split")
    parser.add_argument("--limit-queries", type=int, default=1000, help="Number of query passage clusters to ingest (0 for full dataset)")
    parser.add_argument("--batch-size", type=int, default=128, help="Embedding and upsert batch size")
    parser.add_argument("--language", type=str, default="en", choices=["en", "native", "both"], help="Language to index (English passages, native Indic, or both)")
    parser.add_argument("--recreate-collection", action="store_true", help="Delete and recreate the Qdrant collection before indexing")
    return parser.parse_args()

def query_sparse_vector(text: str) -> tuple[list[int], list[float]]:
    tokens = re.findall(r"\w+", text.lower())
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
    args = parse_args()
    print("=" * 75)
    print("AI4BHARAT / MSMARCO-XI DATASET INGESTION PIPELINE")
    print(f"Target Qdrant: {QDRANT_URL} | Collection: {COLLECTION}")
    print(f"Split: {args.split} | Limit Queries: {args.limit_queries or 'FULL DATASET'} | Language Mode: {args.language}")
    print("=" * 75)

    client = QdrantClient(url=QDRANT_URL)

    if args.recreate_collection:
        if client.collection_exists(COLLECTION):
            print(f"Deleting existing collection '{COLLECTION}'...")
            client.delete_collection(COLLECTION)
        
        print(f"Creating collection '{COLLECTION}' with dense cosine metric...")
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config={"dense": VectorParams(size=384, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": {}}
        )
    elif not client.collection_exists(COLLECTION):
        print(f"Collection '{COLLECTION}' does not exist. Creating...")
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config={"dense": VectorParams(size=384, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": {}}
        )

    print("Initializing local FastEmbed ONNX embedding model (BAAI/bge-small-en-v1.5)...")
    embed_model = TextEmbedding("BAAI/bge-small-en-v1.5")

    print(f"Streaming 'AI4Bharat/MSMARCO-XI' ({args.split} split) from Hugging Face...")
    ds = load_dataset("AI4Bharat/MSMARCO-XI", "default", split=args.split, streaming=True)

    pending_texts = []
    pending_metadata = []
    total_indexed = 0
    query_count = 0
    start_time = time.time()

    for row_idx, row in enumerate(ds):
        eng_q = row.get("Eng_Query", "")
        native_q = row.get("query", "")
        passages_obj = row.get("passages", {})

        eng_passages = passages_obj.get("English_passages", [])
        native_passages = passages_obj.get("passage_text", [])
        is_selected = passages_obj.get("is_selected", [])

        passages_to_add = []
        if args.language in ["en", "both"] and eng_passages:
            for p_idx, p in enumerate(eng_passages):
                if p and str(p).strip():
                    sel = is_selected[p_idx] if p_idx < len(is_selected) else 0
                    passages_to_add.append((str(p).strip(), "en", p_idx, sel))

        if args.language in ["native", "both"] and native_passages:
            for p_idx, p in enumerate(native_passages):
                if p and str(p).strip():
                    sel = is_selected[p_idx] if p_idx < len(is_selected) else 0
                    passages_to_add.append((str(p).strip(), row.get("target_lang", "indic"), p_idx, sel))

        if not passages_to_add:
            continue

        query_count += 1

        for p_text, lang, p_idx, sel in passages_to_add:
            pending_texts.append(p_text)
            pending_metadata.append({
                "text": p_text,
                "language": lang,
                "chunk_index": p_idx,
                "is_selected": sel,
                "query_id": str(row.get("query_id", "")),
                "source_doc": f"{row_idx}_{p_idx}",
                "length": len(p_text)
            })

            # Process in batches
            if len(pending_texts) >= args.batch_size:
                vectors = list(embed_model.embed(pending_texts))
                points = []
                for i, vec in enumerate(vectors):
                    meta = pending_metadata[i]
                    p_txt = pending_texts[i]
                    indices, values = query_sparse_vector(p_txt)
                    sparse_vec = SparseVector(indices=indices, values=values) if indices else None
                    
                    vec_dict = {"dense": vec.tolist()}
                    if sparse_vec:
                        vec_dict["sparse"] = sparse_vec
                        
                    points.append(PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vec_dict,
                        payload=meta
                    ))

                client.upsert(collection_name=COLLECTION, points=points)
                total_indexed += len(points)
                elapsed = time.time() - start_time
                rate = total_indexed / elapsed if elapsed > 0 else 0
                print(f"Indexed {total_indexed} passages ({query_count} queries) | Rate: {rate:.1f} passages/sec")
                pending_texts = []
                pending_metadata = []

        if args.limit_queries > 0 and query_count >= args.limit_queries:
            break

    # Flush remaining batch
    if pending_texts:
        vectors = list(embed_model.embed(pending_texts))
        points = []
        for i, vec in enumerate(vectors):
            meta = pending_metadata[i]
            p_txt = pending_texts[i]
            indices, values = query_sparse_vector(p_txt)
            sparse_vec = SparseVector(indices=indices, values=values) if indices else None
            
            vec_dict = {"dense": vec.tolist()}
            if sparse_vec:
                vec_dict["sparse"] = sparse_vec
                
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=vec_dict,
                payload=meta
            ))
        client.upsert(collection_name=COLLECTION, points=points)
        total_indexed += len(points)

    total_time = time.time() - start_time
    print("=" * 75)
    print(f"🎉 Ingestion Complete! Successfully indexed {total_indexed} passages across {query_count} queries.")
    print(f"⏱️ Total Time: {total_time:.2f}s | Avg Speed: {total_indexed/total_time:.1f} passages/sec")
    print("=" * 75)

if __name__ == "__main__":
    main()
