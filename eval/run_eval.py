"""
CI regression gate. Runs the golden set against whatever prompt is currently labeled 'production', scores each item, and fails (non-zero exit code) if the pass rate drops below an absolute floor.

Calls run_rag_pipeline() directly, in-process - NOT through  the /query HTTP endpoint. Two consequences worth knowing:
    1. No running FastAPI server needed in CI - just the Python process.
    2. This never touches Postgres/request_log(only /query does that), so CI eval runs don't pollute the production drift-detection data. No database needs to run in CI at all.
"""

import json
import re
import sys
from pathlib import Path
from langfuse import get_client
from dotenv import load_dotenv

load_dotenv()

from app.rag_pipeline import run_rag_pipeline

langfuse = get_client()

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.json"
PASS_RATE_FLOOR = 0.83  # 10 OF 12 - tune this as your golden set grows


CITATION_PATTERN = re.compile(r"doc-\d{3}")

REFUSAL_PATTERNS = [
    r"(?:does not|doesn'?t) contain(?:\s+\w+){0,3}\s+information",
    r"cannot (?:\w+\s+){0,2}(?:answer|determine)",
    r"can'?t (?:\w+\s+){0,2}(?:answer|determine)",
    r"context (?:\w+\s+){0,2}(?:doesn'?t|does not) (?:\w+\s+){0,2}(?:provide|contain)",
    r"not (?:enough|sufficient) information",
    r"unable to (?:\w+\s+){0,2}(?:answer|determine)",
    r"no (?:information|mention) (?:of|about)",
]

def cites_source(answer: str) -> bool:
    return bool(CITATION_PATTERN.search(answer))


def is_grounded_refusal(answer: str) -> bool:
    lowered = answer.lower()
    return any(re.search(p, lowered) for p in REFUSAL_PATTERNS)


def evaluate_item(item: dict) -> dict:
    result = run_rag_pipeline(
        query=item["query"], user_id="ci-eval", prompt_version=None
    )
    answer = result["answer"]

    if item["expect_in_scope"]:
        passed = cites_source(answer) and not is_grounded_refusal(answer)
    else:
        passed = is_grounded_refusal(answer)

    return {
        "id": item["id"],
        "query": item["query"],
        "answer": answer,
        "passed": passed,
    }


def main():
    golden_set = json.loads(GOLDEN_SET_PATH.read_text())
    results = [evaluate_item(item) for item in golden_set]

    passed_count = sum(r["passed"] for r in results)
    total = len(results)
    pass_rate = passed_count / total

    print(
        f"\n{'=' * 50} \nGOLDEN SET RESULTS: {passed_count}/{total} ({pass_rate:.1%})\n {'=' * 50}"
    )
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f" [{status}] {r['id']}: {r['query']}")
        if not r["passed"]:
            print(f"        answer:{r['answer'][:150]}")

    langfuse.flush()

    if pass_rate < PASS_RATE_FLOOR:
        print(
            f"\n Pass rate {pass_rate:.1%} is below the floor of {PASS_RATE_FLOOR: .1%} - failing CI"
        )
        sys.exit(1)
    print(f"\n Pass rate {pass_rate:.1%} meets the floor of {PASS_RATE_FLOOR:.1%}")
    sys.exit(0)


if __name__ == "__main__":
    main()
