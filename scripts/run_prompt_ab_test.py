"""
Runs 50 requests against prompt v1 (production baseline) and 50 against v2
(citation variant), scores each, and reports whether the difference in outcome rates is statistically significant.
"""

from dotenv import load_dotenv
import math

load_dotenv()
from langfuse import get_client
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag_pipeline import run_rag_pipeline
from score_answer import score_answer

langfuse_client = get_client()

TEST_QUESTIONS = [
    "What is FastAPI?",
    "What does Pydantic do?",
    "How do Docker containers work?",
    "What is a vector database?",
    "How does RAG work?",
    "What is dependency injection?",
    "What does asyncpg do?",
    "What is a multi-stage Docker build?",
    "What is Recall@K?",
    "What is HNSW?",
    "What is a reverse proxy?",
    "What is Redis used for?",
]
N_PER_VARIANT = 50


def run_variant(prompt_version: int) -> list[dict]:
    results = []
    for i in range(N_PER_VARIANT):
        question = TEST_QUESTIONS[i % len(TEST_QUESTIONS)]
        result = run_rag_pipeline(
            query=question,
            user_id=f"ab-test-v{prompt_version}",
            prompt_version=prompt_version,
        )
        scores = score_answer(result["trace_id"], result["answer"])
        results.append(scores)
        print(
            f" v{prompt_version} [{i + 1}/{N_PER_VARIANT}] cites={scores['cites_source']} len_ok={scores['reasonable_length']}"
        )
    return results


def two_proportion_z_test(x1: int, n1: int, x2: int, n2: int) -> tuple[float, float]:
    """
    Standard two-proportion z-test, implemented from scratch (stdlib only - no scipy dependency for one formula). Returns (z_statistic, two_tailed_p_value).
    """
    p1, p2 = x1 / n1, x2 / n2
    p_pooled = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p_pooled * (1 - p_pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    # Standard normal CDF via the error function (math.erf is stdlib)
    cdf = 0.5 * (1 + math.erf(abs(z) / math.sqrt(2)))
    p_value = 2 * (1 - cdf)
    return z, p_value


def summarize(name: str, results_a: list[dict], results_b: list[dict]):
    x_a = sum(r[name] for r in results_a)
    x_b = sum(r[name] for r in results_b)
    n_a, n_b = len(results_a), len(results_b)
    z, p = two_proportion_z_test(x_a, n_a, x_b, n_b)

    print(f"\n{name}:")
    print(f" Variant A (v1, baseline): {x_a}/{n_a} ({100 * x_a / n_a:.1f}%)")
    print(f" Variant B (v2, citation): {x_b}/{n_b} ({100 * x_b / n_b:.1f}%)")
    print(f" z = {z:.2f}, p = {p:.4f}", end=" ")
    if p < 0.05:
        print("-> statistically significant (p < 0.05)")
    else:
        print(
            "-> NOT significant at this sample size (doesn't mean 'no difference' - may mean n=50 isn't enough to detect it)"
        )


if __name__ == "__main__":
    print("Running Variant A (v1, production baseline)...")
    results_a = run_variant(prompt_version=1)

    print("\n Running Variant B (v2, citation instruction)...")
    results_b = run_variant(prompt_version=2)

    print("\n" + "=" * 60)
    print("PROMPT A/B TEST RESULTS")
    print("=" * 60)
    summarize("cites_source", results_a, results_b)
    summarize("reasonable_length", results_a, results_b)
    langfuse_client.flush()
