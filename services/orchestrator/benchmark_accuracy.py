import asyncio
import json
import time
import websockets
import os
from asyncio import Semaphore

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
from groq import AsyncGroq

async def test_query(ws_url, query):
    start_time = time.time()
    try:
        async with websockets.connect(ws_url) as ws:
            req = {"type": "text_query", "query": query, "tts": False}
            await ws.send(json.dumps(req))
            
            answer = ""
            while True:
                try:
                    msg_str = await asyncio.wait_for(ws.recv(), timeout=10.0)
                    msg = json.loads(msg_str)
                    if msg.get("type") == "generation_chunk":
                        answer += msg.get("text", "")
                    elif msg.get("type") == "generation_result":
                        if not answer:
                            answer = msg.get("answer", "")
                    elif msg.get("type") == "done":
                        break
                except asyncio.TimeoutError:
                    break
            
            latency = time.time() - start_time
            return answer, latency
    except Exception as e:
        return f"Error: {e}", time.time() - start_time

async def process_query(idx, ws_url, query, client, sem):
    async with sem:
        ans, lat = await test_query(ws_url, query)
        is_correct, reason = await check_correctness(query, ans, client)
        print(f"[{idx}] Query: {query[:50]}... | Latency: {lat*1000:.1f}ms | Correct: {is_correct} ({reason})")
        return lat, is_correct

async def check_correctness(query, answer, client):
    if not answer or "I am not aware of that" in answer or "not enough information" in answer.lower():
        return False, "Abstained"
        
    prompt = f"Query: {query}\nAnswer: {answer}\nIs this answer relevant and logically sound for the query? Answer with YES or NO."
    try:
        completion = await client.chat.completions.create(
            model="qwen/qwen3.6-27b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10
        )
        res = completion.choices[0].message.content.strip().upper()
        return "YES" in res, res
    except Exception as e:
        return False, str(e)

async def main():
    queries = []
    with open("src/query_set.jsonl", "r") as f:
        for line in f:
            data = json.loads(line)
            queries.append(data["query"])
            
    # We will test exactly 200 questions (or as many as available up to 200)
    queries = queries[:200]
    print(f"Running benchmark on {len(queries)} questions...")
    
    ws_url = "ws://localhost:8000/ws"
    client = AsyncGroq(api_key=GROQ_API_KEY)
    sem = Semaphore(10)
    
    tasks = [process_query(i+1, ws_url, q, client, sem) for i, q in enumerate(queries)]
    results = await asyncio.gather(*tasks)
    
    total_latency = sum(r[0] for r in results)
    correct_count = sum(1 for r in results if r[1])
        
    avg_latency = (total_latency / len(queries)) * 1000 if queries else 0
    accuracy = (correct_count / len(queries)) * 100 if queries else 0
    
    print("\n" + "="*40)
    print("BENCHMARK RESULTS")
    print(f"Total Queries Tested : {len(queries)}")
    print(f"Average Latency      : {avg_latency:.2f} ms")
    print(f"Answer Accuracy      : {accuracy:.2f}% ({correct_count}/{len(queries)})")
    print("="*40)

if __name__ == "__main__":
    asyncio.run(main())
