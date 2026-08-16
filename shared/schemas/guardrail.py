from pydantic import BaseModel


class ConfidenceScore(BaseModel):
    query: str
    top_score: float
    threshold: float
    is_confident: bool


class AbstentionDecision(BaseModel):
    should_abstain: bool
    reason: str
    confidence: ConfidenceScore
