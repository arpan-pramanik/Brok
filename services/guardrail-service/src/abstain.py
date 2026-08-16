import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "shared"))
from schemas.guardrail import ConfidenceScore, AbstentionDecision

DEFAULT_THRESHOLD = float(os.getenv("ABSTAIN_THRESHOLD", "0.3"))


def should_abstain(
    query: str,
    top_rerank_score: float,
    threshold: float = DEFAULT_THRESHOLD,
) -> AbstentionDecision:
    confidence = ConfidenceScore(
        query=query,
        top_score=top_rerank_score,
        threshold=threshold,
        is_confident=top_rerank_score >= threshold,
    )
    if top_rerank_score < threshold:
        return AbstentionDecision(
            should_abstain=True,
            reason=f"top reranker score {top_rerank_score:.3f} below threshold {threshold:.3f}",
            confidence=confidence,
        )
    return AbstentionDecision(
        should_abstain=False,
        reason="confident",
        confidence=confidence,
    )
