import json
import math
from pathlib import Path
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

client = QdrantClient(url="http://localhost:6333")
model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

K = 5
CHUNK_SIZES = [256, 512, 1024]


def find_rank(hits, expected_snippet: str, expected_doc: str) -> int | None:
    for i, hit in enumerate(hits, start=1):
        payload = hit.payload
        if (
            payload["source_doc"] == expected_doc
            and expected_snippet in payload["text"]
        ):
            return i
    return None


def evaluate(chunk_size: int, golden_set: list[dict]) -> dict:
    collection = f"docs_{chunk_size}"
    hits_found, reciprocal_ranks, ndcgs = 0, [], []

    for item in golden_set:
        vector = model.encode(f"search_document: {item['query']}").tolist()
        result = client.query_points(collection_name=collection, query=vector, limit=K)
        rank = find_rank(result.points, item["expected_snippet"], item["source_doc"])

        if rank:
            hits_found += 1
            reciprocal_ranks.append(1.0 / rank)
            ndcgs.append(1.0 / math.log2(rank + 1))
        else:
            reciprocal_ranks.append(0.0)
            ndcgs.append(0.0)

    n = len(golden_set)
    return {
        "chunk_size": chunk_size,
        f"recall@{K}": hits_found / n,
        "mrr": sum(reciprocal_ranks) / n,
        f"ndcg@{K}": sum(ndcgs) / n,
    }


def main():
    print("********------>>>>>>>", model.max_seq_length)
    golden_set = json.loads(Path("eval/golden/queries.json").read_text())
    results = [evaluate(size, golden_set) for size in CHUNK_SIZES]
    print(f"{'chunk_size':<12}{'recall@5':<12}{'mrr':<10}{'ndcg@5':<10}")
    for r in results:
        print(
            f"{r['chunk_size']:<12}{r['recall@5']:<12.3f}{r['mrr']:<10.3f}{r['ndcg@5']:<10.3f}"
        )

    Path("eval/results_10_2.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
