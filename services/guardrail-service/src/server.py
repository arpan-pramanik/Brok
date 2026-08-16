import os
import sys
import time
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "shared"))

from abstain import should_abstain
from schemas.guardrail import AbstentionDecision

app = FastAPI(title="Bragi Guardrail Service")


class GuardrailRequest(BaseModel):
    query: str
    top_rerank_score: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/check")
def check(req: GuardrailRequest) -> AbstentionDecision:
    return should_abstain(req.query, req.top_rerank_score)
