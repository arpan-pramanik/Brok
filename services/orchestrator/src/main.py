import os
import sys
import json
import asyncio
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))

from pipeline import run_text_pipeline

app = FastAPI(title="Bragi Orchestrator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextQuery(BaseModel):
    query: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/query")
async def query(req: TextQuery):
    result = await run_text_pipeline(req.query)
    return result


@app.post("/api/benchmark")
async def benchmark():
    benchmark_path = Path(__file__).parent.parent.parent.parent / "benchmark" / "query_set.jsonl"
    if not benchmark_path.exists():
        return {"error": "query_set.jsonl not found"}

    queries = []
    with open(benchmark_path) as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))

    results = []
    for q in queries:
        result = await run_text_pipeline(q["query"])
        results.append(result)

    stage_times: dict[str, list[float]] = {}
    total_times = []
    for r in results:
        if r.get("error"):
            continue
        total_times.append(r.get("total_time_ms", 0))
        for s in r.get("stages", []):
            stage_times.setdefault(s["stage"], []).append(s["duration_ms"])

    def percentiles(vals):
        if not vals:
            return {"p50": 0, "p70": 0, "p100": 0}
        vals = sorted(vals)
        n = len(vals)
        return {
            "p50": vals[int(n * 0.5)],
            "p70": vals[int(n * 0.7)],
            "p100": vals[-1],
        }

    summary = {
        "total_queries": len(results),
        "errors": sum(1 for r in results if r.get("error")),
        "abstentions": sum(1 for r in results if r.get("abstained")),
        "total_latency": percentiles(total_times),
        "per_stage": {stage: percentiles(times) for stage, times in stage_times.items()},
    }
    return {"summary": summary, "results": results}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)

            if msg.get("type") == "text_query":
                result = await run_text_pipeline(msg["query"])
                for stage in result.get("stages", []):
                    await ws.send_json({"type": "stage_timing", **stage})

                if result.get("retrieval"):
                    await ws.send_json({
                        "type": "retrieval_result",
                        "query": result["query"],
                        "candidates": result["retrieval"].get("candidates", []),
                        "retrieval_time_ms": result["retrieval"].get("retrieval_time_ms", 0),
                    })

                if result.get("guardrail"):
                    await ws.send_json({
                        "type": "guardrail_result",
                        **result["guardrail"],
                    })

                await ws.send_json({
                    "type": "generation_result",
                    "answer": result.get("answer", ""),
                    "sources": result.get("sources", []),
                    "model_used": result.get("model_used", ""),
                    "generation_time_ms": result.get("generation_time_ms", 0),
                    "fallback_used": result.get("fallback_used", False),
                })

            elif msg.get("type") == "audio_chunk":
                await ws.send_json({
                    "type": "error",
                    "message": "audio streaming requires ASR service connection",
                })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except:
            pass
