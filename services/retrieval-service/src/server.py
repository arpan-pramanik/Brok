import os
import sys
import time
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "shared"))

from hybrid_search import hybrid_search
from rrf_fusion import rrf_fuse
from reranker import rerank
from schemas.retrieval import ChunkCandidate, RRFScore, RetrievalResult

app = FastAPI(title="Bragi Retrieval Service")


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


@app.get("/health")
def health():
    return {"status": "ok"}


from functools import lru_cache

@lru_cache(maxsize=1024)
def _cached_search(query: str, top_k: int):
    dense_results, sparse_results = hybrid_search(query)
    fused = rrf_fuse(dense_results, sparse_results, top_n=6)
    reranked = rerank(query, fused, top_n=top_k)
    return fused, reranked

@app.post("/retrieve")
def retrieve(req: QueryRequest) -> RetrievalResult:
    start = time.time()
    fused, reranked = _cached_search(req.query, req.top_k)

    candidates = [
        ChunkCandidate(
            chunk_id=r.get("chunk_id", str(r["id"])),
            text=r["text"],
            score=r.get("rerank_score", r.get("rrf_score", 0)),
            source_doc=r.get("source_doc", ""),
            chunk_index=r.get("chunk_index", 0),
        )
        for r in reranked
    ]

    rrf_scores = [
        RRFScore(
            chunk_id=r.get("chunk_id", str(r["id"])),
            dense_rank=r.get("dense_rank"),
            sparse_rank=r.get("sparse_rank"),
            fused_score=r.get("rrf_score", 0),
        )
        for r in fused
    ]

    elapsed = (time.time() - start) * 1000
    return RetrievalResult(
        query=req.query,
        candidates=candidates,
        rrf_scores=rrf_scores,
        top_reranked=candidates,
        retrieval_time_ms=elapsed,
    )
