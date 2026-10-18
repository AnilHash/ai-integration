# eval/run_behavioral_tests.py
"""
Ten behavioral regression tests -- hallucination guard, citation accuracy,
adversarial refusal. All heuristically checkable by design (Part A's
"known trap" pattern), so this runs in the fast PR gate.
"""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from app.rag_pipeline import run_rag_pipeline
from eval.run_eval import is_grounded_refusal, CITATION_PATTERN

BEHAVIORAL_SET_PATH = Path("eval/golden/behavioral_tests.json")
MAX_ATTEMPTS = 3


def check_hallucination_guard(answer: str, trap_phrases: list[str]) -> bool:
    lowered = answer.lower()
    return not any(trap.lower() in lowered for trap in trap_phrases)


def citation_matches_source(
    answer: str, citations: list[dict], correct_source_doc: str
) -> bool:
    cited_labels = set(CITATION_PATTERN.findall(answer))
    if not cited_labels:
        return False
    label_to_source = {c["label"]: c["source_doc"] for c in citations}
    return all(
        label_to_source.get(label) == correct_source_doc for label in cited_labels
    )


def check_item(item: dict, result: dict) -> bool:
    answer = result["answer"]
    if item["type"] == "hallucination_guard":
        return check_hallucination_guard(answer, item["trap_phrases"])
    if item["type"] == "citation_accuracy":
        return citation_matches_source(
            answer, result["citations"], item["correct_source_doc"]
        )
    if item["type"] == "refusal_adversarial":
        return is_grounded_refusal(answer)
    raise ValueError(f"unknown behavioral test type: {item['type']}")


def evaluate_item(item: dict, max_attempts: int = MAX_ATTEMPTS) -> dict:
    # Same retry-on-failure shape as eval/run_eval.py's evaluate_item (11.3) --
    # second use of this exact pattern. One more and it's a rule-of-three
    # extraction, same call 11.3 already made for eval/stats.py.
    last_answer = None
    for attempt_num in range(1, max_attempts + 1):
        result = run_rag_pipeline(query=item["query"])
        last_answer = result["answer"]
        if check_item(item, result):
            return {
                "id": item["id"],
                "type": item["type"],
                "passed": True,
                "attempts": attempt_num,
                "flaky": attempt_num > 1,
            }
    return {
        "id": item["id"],
        "type": item["type"],
        "passed": False,
        "attempts": max_attempts,
        "flaky": False,
        "answer": last_answer,
    }


def main():
    items = json.loads(BEHAVIORAL_SET_PATH.read_text())
    results = [evaluate_item(item) for item in items]

    all_passed = True
    by_type = {}
    for r in results:
        by_type.setdefault(r["type"], []).append(r)

    for test_type, group in by_type.items():
        passed = sum(r["passed"] for r in group)
        print(f"{test_type}: {passed}/{len(group)}")
        for r in group:
            status = "PASS" if r["passed"] else "FAIL"
            if not r["passed"]:
                all_passed = False
            print(
                f"  [{status}] {r['id']}"
                + (f" (answer: {r.get('answer', '')[:100]})" if not r["passed"] else "")
            )

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
