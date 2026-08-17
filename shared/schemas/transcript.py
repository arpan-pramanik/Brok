from pydantic import BaseModel, Field
from typing import Optional
import time


class PartialTranscript(BaseModel):
    type: str = "partial_transcript"
    text: str
    is_final: bool = False
    confidence: float = 0.0
    timestamp: float = Field(default_factory=time.time)
    segment_id: int = 0


class FinalTranscript(BaseModel):
    type: str = "final_transcript"
    text: str
    confidence: float
    duration_seconds: float = 0.0
    timestamp: float = Field(default_factory=time.time)
    language: str = "en"
