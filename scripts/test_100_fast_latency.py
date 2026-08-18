import asyncio
import json
import time
import pandas as pd
import websockets

WSS_URL = "ws://localhost:8000/ws"

async def run_100_tests():
    print("==================================================")
    print("⚡ RUNNING 100 QUESTION ULTRA-LOW LATENCY BENCHMARK")
    print("==================================================")

    df = pd.read_parquet("/tmp/hinval.parquet")
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

    print(f"Loaded {len(test_items)} unique questions from MSMARCO-XI.\n")

    ttft_times = []
    total_times = []
    success_count = 0

    for idx, item in enumerate(test_items, 1):
        query_text = item["query"]
        t0 = time.time()
        ttft = None
        full_text = ""
        stages = {}

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
                        elif msg_type == "generation_chunk":
                            if ttft is None:
                                ttft = (time.time() - t0) * 1000
                            full_text += data.get("text", "")
                        elif msg_type == "done":
                            break
                    except asyncio.TimeoutError:
                        break

            elapsed = (time.time() - t0) * 1000
            total_times.append(elapsed)

            if ttft is not None:
                ttft_times.append(ttft)
            else:
                ttft_times.append(elapsed)

            is_success = elapsed < 150.0 or (ttft is not None and ttft < 150.0)
            if is_success:
                success_count += 1

            status_icon = "⚡" if is_success else "⏱️"
            ttft_str = f"{ttft:.1f}ms" if ttft is not None else "N/A"
            print(f"[{idx:03d}/100] {status_icon} TTFT: {ttft_str:>7} | Total: {elapsed:>6.1f}ms | Q: \"{query_text[:45]}\"")

        except Exception as e:
            print(f"[{idx:03d}/100] ❌ Error: {e}")

    ttft_times.sort()
    total_times.sort()

    def get_pct(vals, pct):
        if not vals: return 0.0
        idx = int((len(vals) - 1) * pct)
        return vals[idx]

    print("\n==================================================")
    print("📊 FINAL 100 QUESTION LATENCY BENCHMARK RESULTS")
    print("==================================================")
    print(f"Total Questions Benchmark     : {len(test_items)}")
    print(f"Under 150ms Target Pass Rate  : {success_count}/{len(test_items)} ({success_count / len(test_items) * 100:.1f}%)\n")
    print("Time to First Token (TTFT - Perceived User Latency):")
    print(f"  - TTFT P50  : {get_pct(ttft_times, 0.50):.1f} ms")
    print(f"  - TTFT P70  : {get_pct(ttft_times, 0.70):.1f} ms")
    print(f"  - TTFT P100 : {get_pct(ttft_times, 1.00):.1f} ms\n")
    print("Total Pipeline Completion Time:")
    print(f"  - Total P50 : {get_pct(total_times, 0.50):.1f} ms")
    print(f"  - Total P70 : {get_pct(total_times, 0.70):.1f} ms")
    print(f"  - Total P100: {get_pct(total_times, 1.00):.1f} ms")
    print("==================================================")

if __name__ == "__main__":
    asyncio.run(run_100_tests())
