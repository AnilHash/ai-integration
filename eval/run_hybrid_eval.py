import json
import math
from pathlib import Path
from app.hybrid_retrieval import (
    dense_search,
    bm25_search,
    rerank,
    reciprocal_rank_fusion,
    payload_lookup,
)

K = 5


def find_rank(
    candidate_ids: list[int], expected_snippet: str, expected_doc: str
) -> int | None:
    for i, cid in enumerate(candidate_ids, start=1):
        payload = payload_lookup[cid]
        if (
            payload["source_doc"] == expected_doc
            and expected_snippet in payload["text"]
        ):
            return i
    return None


def score_method(method_name: str, retrieve_fn, golden_set: list[dict]) -> dict:
    hits, rr, ndcgs = 0, [], []
    for item in golden_set:
        candidate_ids = retrieve_fn(item["query"])
        rank = find_rank(candidate_ids, item["expected_snippet"], item["source_doc"])
        if rank:
            hits += 1
            rr.append(1.0 / rank)
            ndcgs.append(1.0 / math.log2(rank + 1))
        else:
            rr.append(0.0)
            ndcgs.append(0.0)

    n = len(golden_set)
    return {
        "method": method_name,
        f"recall@{K}": hits / n,
        "mrr": sum(rr) / n,
        f"ndcg@{K}": sum(ndcgs) / n,
    }


def dense_only(query: str) -> list[int]:
    return [cid for cid, _ in dense_search(query, k=K)]


def rrf_only(query: str) -> list[int]:
    fused = reciprocal_rank_fusion(dense_search(query, k=20), bm25_search(query, k=20))
    return [cid for cid, _ in fused[:K]]


def rrf_reranked(query: str) -> list[int]:
    fused = reciprocal_rank_fusion(dense_search(query, k=20), bm25_search(query, k=20))
    candidate_ids = [cid for cid, _ in fused[:20]]
    result = rerank(query, candidate_ids, top_k=K)
    return [item["id"] for item in result]


def main():
    golden_set = json.loads(Path("eval/golden/queries.json").read_text())
    results = [
        score_method("dense-only", dense_only, golden_set),
        score_method("hybrid-rrf", rrf_only, golden_set),
        score_method("hybrid-rrf-reranked", rrf_reranked, golden_set),
    ]
    print(f"{'method':<22}{'recall@5':<12}{'mrr':<10}{'ndcg@5':<10}")
    for r in results:
        print(
            f"{r['method']:<22}{r['recall@5']:<12.3f}{r['mrr']:<10.3f}{r['ndcg@5']:<10.3f}"
        )
    Path("eval/results_10_3.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
