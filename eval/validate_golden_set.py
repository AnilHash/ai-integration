import json
import sys
from pathlib import Path


REQUIRED_BASE = {"id", "query", "category", "expect_in_scope", "tiers_applicable"}


def validate(path: str = "eval/golden/golden_50.json") -> list[str]:
    items = json.loads(Path(path).read_text())

    errors = []
    seen_ids = set()

    for item in items:
        missing = REQUIRED_BASE - item.keys()
        if missing:
            errors.append(f"{item.get('id', '?')}: missing base fields {missing}")
        if item["id"] in seen_ids:
            errors.append(f"{item['id']}: duplicate id")
        seen_ids.add(item["id"])

        if item["expect_in_scope"]:
            if "retrieval" not in item or "reference_answer" not in item:
                errors.append(
                    f"{item['id']}: in-scope item missing retrieval/reference_answer"
                )
            if "judge" not in item.get("tiers_applicable", []):
                errors.append(
                    f"{item['id']}: in_scope item should apply the judge tier"
                )
        else:
            if "retrieval" in item or " reference_answer" in item:
                errors.append(
                    f"{item['id']}: out-of-scope item shouldn't carry retrieval/reference fields"
                )

    return errors


if __name__ == "__main__":
    errors = validate()
    if errors:
        print(f"{len(errors)} problem(s):")
        for e in errors:
            print(f" - {e}")
        sys.exit(1)
    print("Golden set structurally valid.")
