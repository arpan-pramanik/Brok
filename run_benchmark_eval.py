import asyncio
import json
import time
import httpx

ORCH_URL = "http://localhost:8000"

# 10 in-context + 10 out-of-context = 20 queries
TEST_QUERIES = [
    # In-context (MSMARCO-XI dataset)
    {"query": "What is artificial intelligence?", "expected": "answer"},
    {"query": "What was the Manhattan Project?", "expected": "answer"},
    {"query": "What is photosynthesis?", "expected": "answer"},
    {"query": "What are the goals of AI research?", "expected": "answer"},
    {"query": "When was the first nuclear weapon produced?", "expected": "answer"},
    {"query": "Who led the Manhattan Project?", "expected": "answer"},
    {"query": "What is machine learning?", "expected": "answer"},
    {"query": "What does photosynthesis convert?", "expected": "answer"},
    {"query": "What is natural language processing?", "expected": "answer"},
    {"query": "What is knowledge representation?", "expected": "abstain"},
    # Out-of-context (should abstain)
    {"query": "Which video game features Spiderman?", "expected": "abstain"},
    {"query": "How to cook Italian pasta carbonara?", "expected": "abstain"},
    {"query": "What is the current stock price of Apple?", "expected": "abstain"},
    {"query": "How to learn Japanese language fast?", "expected": "abstain"},
    {"query": "Explain general relativity in detail", "expected": "abstain"},
    {"query": "What is the best smartphone in 2024?", "expected": "abstain"},
    {"query": "How to fix a flat tire?", "expected": "abstain"},
    {"query": "Who won the 2022 FIFA World Cup?", "expected": "abstain"},
    {"query": "What is the recipe for chocolate cake?", "expected": "abstain"},
    {"query": "How tall is the Eiffel Tower?", "expected": "abstain"},
]

NUM_RUNS = 3

def pct(vals):
    if not vals:
        return {"p50": 0, "p70": 0, "p100": 0, "avg": 0, "min": 0}
    s = sorted(vals)
    n = len(s)
    return {
        "min": round(s[0], 2),
        "p50": round(s[int(n * 0.50)], 2),
        "p70": round(s[int(n * 0.70)], 2),
        "p100": round(s[-1], 2),
        "avg": round(sum(s) / n, 2),
    }

async def run_single_query(client, query_text):
    start = time.time()
    resp = await client.post(f"{ORCH_URL}/api/query", json={"query": query_text})
    wall_ms = (time.time() - start) * 1000
    if resp.status_code != 200:
        return None
    data = resp.json()
    stages = data.get("stages", [])
    return {
        "answer": data.get("answer", ""),
        "wall_ms": wall_ms,
        "total_ms": data.get("total_time_ms", wall_ms),
        "retrieval_ms": next((s["duration_ms"] for s in stages if s["stage"] == "retrieval"), 0),
        "guardrail_ms": next((s["duration_ms"] for s in stages if s["stage"] == "guardrail"), 0),
        "generation_ms": next((s["duration_ms"] for s in stages if s["stage"] == "generation"), 0),
    }

async def main():
    print(f"{'='*60}")
    print(f"  BRAGI RAG PIPELINE — FULL BENCHMARK ({NUM_RUNS} runs × {len(TEST_QUERIES)} queries)")
    print(f"{'='*60}\n")

    async with httpx.AsyncClient(timeout=60.0) as client:
        # Warmup
        print("Warming up (2 queries)...")
        await run_single_query(client, "warmup query one")
        await run_single_query(client, "What is AI?")
        print("Warmup done.\n")

        all_totals = []
        all_retrievals = []
        all_guardrails = []
        all_generations = []
        correct = 0
        total_tested = 0
        details = []

        for run_idx in range(NUM_RUNS):
            print(f"--- Run {run_idx + 1}/{NUM_RUNS} ---")
            for item in TEST_QUERIES:
                r = await run_single_query(client, item["query"])
                if r is None:
                    continue
                total_tested += 1
                ans_lower = r["answer"].lower()
                abstained = "sorry" in ans_lower or "dont have" in ans_lower or "don't have" in ans_lower or "i don" in ans_lower
                if (item["expected"] == "abstain" and abstained) or (item["expected"] == "answer" and not abstained):
                    correct += 1
                    verdict = "OK"
                else:
                    verdict = "MISS"

                all_totals.append(r["total_ms"])
                all_retrievals.append(r["retrieval_ms"])
                all_guardrails.append(r["guardrail_ms"])
                all_generations.append(r["generation_ms"])

                print(f"  [{verdict}] {r['total_ms']:7.2f}ms  ret={r['retrieval_ms']:6.2f}  guard={r['guardrail_ms']:5.2f}  gen={r['generation_ms']:6.2f}  | {item['query'][:50]}")
                details.append({**item, **r, "verdict": verdict, "run": run_idx + 1})
            print()

        accuracy = correct / total_tested * 100 if total_tested else 0
        under_150 = sum(1 for t in all_totals if t < 150) / len(all_totals) * 100 if all_totals else 0

        summary = {
            "total_queries_executed": total_tested,
            "guardrail_accuracy": f"{accuracy:.1f}%",
            "pct_under_150ms": f"{under_150:.1f}%",
            "latency_target_met (<150ms P100)": all(t < 150 for t in all_totals),
            "total_pipeline_ms": pct(all_totals),
            "retrieval_stage_ms": pct(all_retrievals),
            "guardrail_stage_ms": pct(all_guardrails),
            "generation_stage_ms": pct(all_generations),
        }

        print(f"{'='*60}")
        print(f"                   FINAL RESULTS")
        print(f"{'='*60}")
        print(json.dumps(summary, indent=2))

        with open("benchmark_results.json", "w") as f:
            json.dump({"summary": summary, "details": details}, f, indent=2)
        print("\nSaved to benchmark_results.json")

if __name__ == "__main__":
    asyncio.run(main())
