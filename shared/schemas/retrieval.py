from pydantic import BaseModel, Field
from typing import Optional


class ChunkCandidate(BaseModel):
    chunk_id: str
    text: str
    score: float
    source_doc: str
    chunk_index: int
    metadata: dict = Field(default_factory=dict)


class RRFScore(BaseModel):
    chunk_id: str
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    fused_score: float


class RetrievalResult(BaseModel):
    query: str
    candidates: list[ChunkCandidate]
    rrf_scores: list[RRFScore]
    top_reranked: list[ChunkCandidate]
    retrieval_time_ms: float
