def rrf_fuse(
    dense_results: list[dict],
    sparse_results: list[dict],
    k: int = 60,
    top_n: int = 15,
) -> list[dict]:
    scores: dict[int, float] = {}
    meta: dict[int, dict] = {}

    for rank, r in enumerate(dense_results):
        rid = r["id"]
        scores[rid] = scores.get(rid, 0) + 1.0 / (k + rank + 1)
        meta[rid] = r

    for rank, r in enumerate(sparse_results):
        rid = r["id"]
        scores[rid] = scores.get(rid, 0) + 1.0 / (k + rank + 1)
        if rid not in meta:
            meta[rid] = r

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [
        {**meta[rid], "rrf_score": score, "dense_rank": _find_rank(dense_results, rid), "sparse_rank": _find_rank(sparse_results, rid)}
        for rid, score in ranked
    ]


def _find_rank(results: list[dict], rid: int) -> int | None:
    for i, r in enumerate(results):
        if r["id"] == rid:
            return i + 1
    return None
