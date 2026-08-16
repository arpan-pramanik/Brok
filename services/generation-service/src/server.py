from fastapi import FastAPI, HTTPException
import sys
import os

# Add parent dir to path to import shared schemas if needed, assuming run from Bragi root or python path set
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from shared.schemas.generation import GenerationRequest, GenerationResponse
from circuit_breaker import CircuitBreakerLLM

app = FastAPI(title="Generation Service")
llm = CircuitBreakerLLM()

@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    try:
        return llm.generate(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}
