import asyncio
import json
import time
import httpx
from pathlib import Path

ORCHESTRATOR_URL = "http://localhost:8000"
QUERY_SET = Path(__file__).parent / "query_set.jsonl"
CONCURRENCY = 5


async def replay(concurrency: int = CONCURRENCY):
    queries = []
    with open(QUERY_SET) as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))

    semaphore = asyncio.Semaphore(concurrency)
    results = []

    async def run_query(q):
        async with semaphore:
            async with httpx.AsyncClient(timeout=120.0) as client:
                start = time.time()
                try:
                    resp = await client.post(
                        f"{ORCHESTRATOR_URL}/api/query",
                        json={"query": q["query"]},
                    )
                    data = resp.json()
                    data["benchmark_time_ms"] = (time.time() - start) * 1000
                    data["category"] = q.get("category", "unknown")
                    return data
                except Exception as e:
                    return {
                        "query": q["query"],
                        "error": str(e),
                        "benchmark_time_ms": (time.time() - start) * 1000,
                        "category": q.get("category", "unknown"),
                    }

    tasks = [run_query(q) for q in queries]
    results = await asyncio.gather(*tasks)
    return list(results)


if __name__ == "__main__":
    results = asyncio.run(replay())
    print(json.dumps(results, indent=2))
