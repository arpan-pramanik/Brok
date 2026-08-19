import os
import json
from datasets import load_dataset

def main():
    print("Streaming AI4Bharat/MSMARCO-XI to extract queries...")
    ds = load_dataset("AI4Bharat/MSMARCO-XI", "default", split="validation", streaming=True)
    
    queries = []
    for item in ds:
        q = item.get("query")
        if q:
            queries.append(q)
        if len(queries) >= 200:
            break
            
    out_path = "/home/arpan/Developments/Bragi/services/orchestrator/src/query_set.jsonl"
    with open(out_path, "w") as f:
        for q in queries:
            f.write(json.dumps({"query": q, "category": "answerable"}) + "\n")
            
    print(f"Wrote {len(queries)} queries to {out_path}.")

if __name__ == "__main__":
    main()
