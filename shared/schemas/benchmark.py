from pydantic import BaseModel, Field
from typing import Optional
import time


class StageTiming(BaseModel):
    stage: str
    start_time: float
    end_time: float
    duration_ms: float

    @classmethod
    def from_interval(cls, stage: str, start: float, end: float):
        return cls(stage=stage, start_time=start, end_time=end, duration_ms=(end - start) * 1000)


class QueryResult(BaseModel):
    query_id: str
    query_text: str
    stages: list[StageTiming]
    total_time_ms: float
    answer: str
    abstained: bool = False
    fallback_used: bool = False
    timestamp: float = Field(default_factory=time.time)
    error: Optional[str] = None
