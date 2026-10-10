from app.rag_pipeline import run_rag_pipeline
from eval.llm_judge import judge_answer_safe

import sys

sys.stdout.reconfigure(encoding="utf-8",errors="replace")

def check_consistency(query, context, reference, candidate, runs=3):
    verdicts = [
        judge_answer_safe(query, context, reference, candidate) for _ in range(runs)
    ]
    agree = all(
        (v.faithful, v.relevant, v.complete)
        == (verdicts[0].faithful, verdicts[0].relevant, verdicts[0].complete)
        for v in verdicts
    )
    print(f"Agreement across {runs} runs: {'YES' if agree else 'NO'}")
    for i, v in enumerate(verdicts, 1):
        print(
            f" run {i}: faithful={v.faithful} relevant={v.relevant} complete={v.complete}"
        )
    return agree


reference_answer = "HNSW builds a multi-layer graph. Each stored vector is a node. Higher layers are sparse, with long-range connections that let a search jump across large distances in the vector space quickly. Lower layers are dense, with short-range connections for fine-grained precision. Search starts at an entry point in the top layer, greedily walks toward the query at each layer, then descends — refining the search as it goes — until it reaches the bottom layer, which contains every vector"

if __name__ == "__main__":
    query = "What is HNSW and how does it work?"

    res = run_rag_pipeline(query, prompt_version=1)
    print(res)
    final_result = check_consistency(
        query,
        context=res["context"],
        reference=reference_answer,
        candidate=res["answer"],
    )
    print(final_result)
