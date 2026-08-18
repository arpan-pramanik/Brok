import asyncio
import json
import time
import pandas as pd
import websockets

WSS_URL = "ws://localhost:8000/ws"

async def run_100_tests():
    print("==================================================")
    print("🧪 RUNNING 100 QUESTION TEST SUITE ON MSMARCO-XI")
    print("==================================================")

    url = "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/validation/hinval.parquet"
    df = pd.read_parquet(url)
    print(f"Loaded {len(df)} total questions from MSMARCO-XI dataset.")

    test_items = []
    seen = set()
    for _, row in df.iterrows():
        q = row.get("Eng_Query", "")
        ans = row.get("Eng_Answer", "")
        if q and q not in seen:
            seen.add(q)
            test_items.append({"query": q, "expected": ans})
        if len(test_items) >= 100:
            break

    print(f"Collected {len(test_items)} test queries.\n")

    results = []
    success_retrieval_count = 0
    total_latency_ms = []

    for idx, item in enumerate(test_items, 1):
        query_text = item["query"]
        t0 = time.time()
        full_text = ""
        stages = {}
        retrieval_data = {}

        try:
            async with websockets.connect(WSS_URL) as ws:
                await ws.send(json.dumps({"type": "text_query", "query": query_text, "tts": False}))

                while True:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=10.0)
                        data = json.loads(msg)
                        msg_type = data.get("type")

                        if msg_type == "stage_timing":
                            stages[data.get("stage")] = data.get("duration_ms")
                        elif msg_type == "retrieval_result":
                            retrieval_data = data
                        elif msg_type == "generation_chunk":
                            full_text += data.get("text", "")
                        elif msg_type == "done":
                            break
                    except asyncio.TimeoutError:
                        break

            elapsed = (time.time() - t0) * 1000
            total_latency_ms.append(elapsed)

            candidates = retrieval_data.get("candidates", [])
            top_score = candidates[0].get("score", 0.0) if candidates else 0.0
            retrieved_ok = len(candidates) > 0 and top_score >= 0.52

            if retrieved_ok:
                success_retrieval_count += 1

            status_icon = "✅" if retrieved_ok else "⚠️"
            print(f"[{idx:03d}/100] {status_icon} Score: {top_score:.3f} | Latency: {elapsed:.1f}ms | Q: \"{query_text[:50]}\"")
            if idx <= 10 or idx % 10 == 0:
                print(f"       Ans: \"{full_text.strip()[:100]}\"")

            results.append({
                "idx": idx,
                "query": query_text,
                "top_score": top_score,
                "candidates_count": len(candidates),
                "answer": full_text.strip(),
                "latency_ms": elapsed,
                "retrieved_ok": retrieved_ok
            })
        except Exception as e:
            print(f"[{idx:03d}/100] ❌ Error: {e}")

    print("\n==================================================")
    print("📊 FINAL 100 QUESTION TEST RESULTS SUMMARY")
    print("==================================================")
    print(f"Total Queries Evaluated     : {len(test_items)}")
    print(f"Successful Context Retrieves: {success_retrieval_count}/{len(test_items)} ({success_retrieval_count / len(test_items) * 100:.1f}%)")

    total_latency_ms.sort()
    p50 = total_latency_ms[int(len(total_latency_ms) * 0.50)]
    p70 = total_latency_ms[int(len(total_latency_ms) * 0.70)]
    p100 = total_latency_ms[-1]

    print(f"Latency P50                  : {p50:.1f} ms")
    print(f"Latency P70                  : {p70:.1f} ms")
    print(f"Latency P100                 : {p100:.1f} ms")
    print("==================================================")

if __name__ == "__main__":
    asyncio.run(run_100_tests())
