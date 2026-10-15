import json
import sys
from pathlib import Path


from dotenv import load_dotenv

load_dotenv()


from eval.retrieval_scorer import score_retrieval


GOLDEN_50_PATH = Path("eval/golden/golden_50.json")
RECALL_FLOOR = 0.5


def main():
    items = json.loads(GOLDEN_50_PATH.read_text(encoding="utf-8"))
    retrieval_items = [i for i in items if "retrieval" in i.get("tiers_applicable", [])]

    hits = 0

    for item in retrieval_items:
        rank = score_retrieval(
            item["query"],
            item["retrieval"]["expected_snippet"],
            item["retrieval"]["source_doc"],
        )
        hit = rank is not None
        hits += hit
        print(f" [{'HIT' if hit else 'MISS'} ] {item['id']}: {item['query']}")

    recall = hits / len(retrieval_items)
    print(f"\n Recall@5: {hits}/{len(retrieval_items)} ({recall:.1%})")

    if recall < RECALL_FLOOR:
        print(f" X Recall@5 {recall:.1%} below floor {RECALL_FLOOR:.1%} - failing CI")
        sys.exit(1)

    print(f" Recall@5 {recall:.1%} meets floor {RECALL_FLOOR:.1%}")


if __name__ == "__main__":
    main()
