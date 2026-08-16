import json
import sys
from pathlib import Path
from sklearn.metrics import f1_score
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "shared"))


def load_dev_set(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def calibrate(dev_set_path: str, rerank_scores: list[float], labels: list[bool]) -> float:
    thresholds = np.arange(0.0, 1.0, 0.01)
    best_t, best_f1 = 0.3, 0.0

    for t in thresholds:
        preds = [score >= t for score in rerank_scores]
        f = f1_score(labels, preds)
        if f > best_f1:
            best_f1 = f
            best_t = float(t)

    print(f"optimal threshold: {best_t:.2f} (F1: {best_f1:.3f})")
    return best_t


if __name__ == "__main__":
    dev_path = str(Path(__file__).parent / "dev_set.jsonl")
    data = load_dev_set(dev_path)
    scores = [d["rerank_score"] for d in data]
    labels = [d["answerable"] for d in data]
    calibrate(dev_path, scores, labels)
