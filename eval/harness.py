import hashlib
import math
import time
import uuid
import json
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from langfuse import get_client

from app.db import get_pool, init_eval_tables
from app.llm_client import DEFAULT_MODEL
from app.hybrid_retrieval import payload_lookup
from app.rag_pipeline import run_rag_pipeline
from eval.llm_judge import judge_answer_safe, JUDGE_MODEL
from eval.run_eval import cites_source, is_grounded_refusal

langfuse = get_client()

GOLDEN_SET_PATH = Path("eval/golden/golden_50.json")
GENERATOR_MODEL = DEFAULT_MODEL
PAYLOADS = {str(k): v for k, v in payload_lookup.items()}


def dataset_version() -> str:
    return hashlib.sha256(GOLDEN_SET_PATH.read_bytes()).hexdigest()[:12]


def rank_from_sources(
    sources: list[str], expected_snippet: str, expected_doc: str
) -> int | None:
    """1-indexed rank of the first chunk the PIPELINE returned (hybrid + rerank)
    that matches both the expected source doc and snippet, else None."""
    for i, chunk_id in enumerate(sources, start=1):
        payload = PAYLOADS.get(str(chunk_id))
        if (
            payload
            and payload["source_doc"] == expected_doc
            and expected_snippet in payload["text"]
        ):
            return i
    return None


def phase_1_generate(golden_set: list[dict]) -> list[dict]:
    """Every generation call, before a single judge call happens."""
    results = []
    for n, item in enumerate(golden_set, start=1):
        start = time.perf_counter()
        try:
            out = run_rag_pipeline(query=item["query"], user_id="eval")
            latency_ms = int((time.perf_counter() - start) * 1000)
            results.append(
                {
                    **item,
                    "answer": out["answer"],
                    "context": out["context"],
                    "sources": out["sources"],
                    "latency_ms": latency_ms,
                    "error": None,
                }
            )
        except Exception as e:
            results.append(
                {
                    **item,
                    "answer": None,
                    "context": "",
                    "sources": [],
                    "latency_ms": None,
                    "error": str(e),
                }
            )
        print(f"  generated {n}/{len(golden_set)}")
    return results


def phase_2_heuristic(results: list[dict]) -> None:
    for r in results:
        if r["error"] or "heuristic" not in r["tiers_applicable"]:
            continue
        r["heuristic_cites_source"] = cites_source(r["answer"])
        r["heuristic_grounded_refusal"] = is_grounded_refusal(r["answer"])


def phase_3_retrieval(results: list[dict]) -> None:
    """Scores what the pipeline actually retrieved; no second embedding call."""
    for r in results:
        if r["error"] or "retrieval" not in r["tiers_applicable"]:
            continue
        r["retrieval_rank"] = rank_from_sources(
            r["sources"],
            r["retrieval"]["expected_snippet"],
            r["retrieval"]["source_doc"],
        )


def phase_4_judge(results: list[dict]) -> None:
    """Runs LAST and separately."""
    for r in results:
        if r["error"] or "judge" not in r["tiers_applicable"]:
            continue
        try:
            verdict = judge_answer_safe(
                r["query"], r["context"], r["reference_answer"], r["answer"]
            )
        except Exception as e:
            r["judge_error"] = str(e)
            continue
        r["judge_faithful"] = verdict.faithful
        r["judge_relevant"] = verdict.relevant
        r["judge_complete"] = verdict.complete


def heuristic_pass(r: dict) -> bool | None:
    """None = not applicable. Generation errors count as failures."""
    if "heuristic" not in r["tiers_applicable"]:
        return None
    if r["error"]:
        return False
    if r["expect_in_scope"]:
        # Same rule as the CI gate in eval/run_eval.py
        return r["heuristic_cites_source"] and not r["heuristic_grounded_refusal"]
    return r["heuristic_grounded_refusal"]


def judge_pass(r: dict) -> bool | None:
    """None = not applicable, or the judge itself failed (excluded, reported
    separately). Generation errors count as failures."""
    if "judge" not in r["tiers_applicable"]:
        return None
    if r["error"]:
        return False
    if r.get("judge_error"):
        return None
    return bool(
        r.get("judge_faithful") and r.get("judge_relevant") and r.get("judge_complete")
    )


def aggregate(results: list[dict]) -> dict:
    h = [v for r in results if (v := heuristic_pass(r)) is not None]
    j = [v for r in results if (v := judge_pass(r)) is not None]
    # Errored items stay in the denominator as misses (rank None).
    ranks = [
        None if r["error"] else r.get("retrieval_rank")
        for r in results
        if "retrieval" in r["tiers_applicable"]
    ]
    n_err = sum(1 for r in results if r["error"])
    n_judge_err = sum(1 for r in results if r.get("judge_error"))
    return {
        "heuristic_pass_rate": sum(h) / len(h) if h else None,
        "retrieval_recall_at_5": (sum(1 for rk in ranks if rk is not None) / len(ranks))
        if ranks
        else None,
        "retrieval_mrr": (sum((1.0 / rk) if rk else 0.0 for rk in ranks) / len(ranks))
        if ranks
        else None,
        "retrieval_ndcg_at_5": (
            sum((1.0 / math.log2(rk + 1)) if rk else 0.0 for rk in ranks) / len(ranks)
        )
        if ranks
        else None,
        "judge_pass_rate": sum(j) / len(j) if j else None,
        # Not stored in eval_runs (no columns); printed only.
        "generation_error_rate": n_err / len(results) if results else None,
        "judge_error_count": n_judge_err,
    }


STORED_AGG_KEYS = (
    "heuristic_pass_rate",
    "retrieval_recall_at_5",
    "retrieval_mrr",
    "retrieval_ndcg_at_5",
    "judge_pass_rate",
)


async def store_run(results: list[dict], agg: dict, started_at: datetime) -> uuid.UUID:
    pool = await get_pool()
    run_id = uuid.uuid4()
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                """
                INSERT INTO eval_runs
                    (run_id, dataset_version, generator_model, judge_model, total_items,
                     heuristic_pass_rate, retrieval_recall_at_5, retrieval_mrr, retrieval_ndcg_at_5,
                     judge_pass_rate, started_at, finished_at)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
                """,
                run_id,
                dataset_version(),
                GENERATOR_MODEL,
                JUDGE_MODEL,
                len(results),
                *(agg[k] for k in STORED_AGG_KEYS),
                started_at,
                datetime.now(timezone.utc),
            )
            for r in results:
                error = r["error"] or (
                    f"judge: {r['judge_error']}" if r.get("judge_error") else None
                )
                await conn.execute(
                    """
                    INSERT INTO eval_results
                        (run_id, item_id, category, heuristic_cites_source, heuristic_grounded_refusal,
                         retrieval_rank, judge_faithful, judge_relevant, judge_complete,
                         latency_ms, raw_answer, error)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
                    """,
                    run_id,
                    r["id"],
                    r["category"],
                    r.get("heuristic_cites_source"),
                    r.get("heuristic_grounded_refusal"),
                    r.get("retrieval_rank"),
                    r.get("judge_faithful"),
                    r.get("judge_relevant"),
                    r.get("judge_complete"),
                    r["latency_ms"],
                    r["answer"],
                    error,
                )
    return run_id


async def main():
    await init_eval_tables()
    golden_set = json.loads(GOLDEN_SET_PATH.read_text())
    started_at = datetime.now(timezone.utc)

    results = phase_1_generate(golden_set)
    print("phase 1 DONE !!")
    phase_2_heuristic(results)
    print("phase 2 DONE !!")
    phase_3_retrieval(results)
    print("phase 3 DONE !!")
    phase_4_judge(results)
    print("phase 4 DONE !!")

    agg = aggregate(results)
    run_id = await store_run(results, agg, started_at)
    langfuse.flush()

    print(f"run_id: {run_id}")
    for k, v in agg.items():
        if v is None:
            print(f"  {k}: n/a")
        elif isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
