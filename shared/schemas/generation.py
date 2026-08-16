from pydantic import BaseModel, Field
from typing import Optional
import time


class GenerationRequest(BaseModel):
    query: str
    context_chunks: list[str]
    max_tokens: int = 512
    temperature: float = 0.3
    source_docs: list[str] = Field(default_factory=list)


class GenerationResponse(BaseModel):
    answer: str
    sources: list[str]
    model_used: str
    generation_time_ms: float
    fallback_used: bool = False
    timestamp: float = Field(default_factory=time.time)
