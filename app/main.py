from datetime import datetime, timedelta, timezone
import json
from statistics import mean, median
import statistics

from dotenv import load_dotenv

load_dotenv()
import langfuse

from app.db import get_pool, init_request_log_table
from score_answer import score_answer
from scripts.run_prompt_ab_test import two_proportion_z_test


from contextlib import asynccontextmanager
from fastapi import FastAPI, Query as QueryParam
from app.instrumentation import verify_langfuse_connection
from app.rag_pipeline import run_rag_pipeline
from app.z_test_means import two_sample_z_test_means

langfuse_client = langfuse.get_client()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not verify_langfuse_connection():
        raise RuntimeError(
            "Langfuse auth_check() failed - traces will not be recorded. "
            ""
            "Check env variables."
        )
    await init_request_log_table()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/query")
async def query_endpoint(
    q: str = QueryParam(..., min_length=3, description="User query string"),
    user_id: str = QueryParam(
        default="anonymous", description="User identifier for cost attribution"
    ),
    prompt_version: int | None = QueryParam(
        default=None,
        description="Force a specific prompt version (1 or 2). Omit to use the production-labelled version.",
    ),
):
    result = run_rag_pipeline(query=q, user_id=user_id, prompt_version=prompt_version)
    scores = score_answer(result["trace_id"], result["answer"])

    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO request_log
                (trace_id, user_id, query, query_length, answer_length, cites_source, reasonable_length, prompt_version)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
            result["trace_id"],
            user_id,
            q,
            len(q),
            len(result["answer"]),
            scores["cites_source"],
            scores["reasonable_length"],
            str(prompt_version) if prompt_version else "production",
        )
    return result


@app.get("/admin/cost-report")
def cost_report(
    hours: int = QueryParam(
        default=24, ge=1, le=720, description="Lookback window in hours"
    ),
):
    """
    Per-user cost/token/request breakdown over the last `hours` hours.
    Uses Metrics API v1 (not v2) because v2 cannot group by userId -
    """
    now = datetime.now(timezone.utc)
    from_ts = (now - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
    to_ts = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    query = json.dumps(
        {
            "view": "traces",
            "metrics": [
                {"measure": "totalCost", "aggregation": "sum"},
                {"measure": "totalTokens", "aggregation": "sum"},
                {"measure": "count", "aggregation": "count"},
            ],
            "dimensions": [{"field": "userId"}],
            "filters": [],
            "fromTimestamp": from_ts,
            "toTimestamp": to_ts,
            "orderBy": [{"field": "sum_totalCost", "direction": "desc"}],
        }
    )
    result = langfuse_client.api.legacy.metrics_v1.metrics(query=query)
    rows = [
        {
            "user_id": row.get("userId", "unknown"),
            "request_count": int(row.get("count_count", 0)),
            "total_tokens": int(row.get("sum_totalTokens", 0)),
            "total_cost_usd": round(float(row.get("sum_totalCost", 0)), 6),
        }
        for row in result.data
    ]
    return {
        "window_hours": hours,
        "from": from_ts,
        "to": to_ts,
        "users": rows,
        "grand_total_cost_usd": round(sum(r["total_cost_usd"] for r in rows), 6),
        "grand_total_requests": sum(r["request_count"] for r in rows),
    }


@app.get("/admin/latency-report")
def latency_report(hours: int = QueryParam(default=24, ge=1, le=720)):
    """
    Per-stage latency percentiles (P50,P95,P99) grouped by observation name.
    Uses Metrics API v2 - grouping by `name` is allowed; grouping by
    userId/traceId is not"""
    now = datetime.now(timezone.utc)
    from_ts = (now - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
    to_ts = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    query = json.dumps(
        {
            "view": "observations",
            "metrics": [
                {"measure": "latency", "aggregation": "avg"},
                {"measure": "latency", "aggregation": "p50"},
                {"measure": "latency", "aggregation": "p95"},
                {"measure": "latency", "aggregation": "p99"},
                {"measure": "count", "aggregation": "count"},
            ],
            "dimensions": [{"field": "name"}],
            "filters": [],
            "fromTimestamp": from_ts,
            "toTimestamp": to_ts,
            "orderBy": [{"field": "p95_latency", "direction": "desc"}],
        }
    )
    result = langfuse_client.api.metrics.metrics(query=query)

    stages = [
        {
            "stage": row.get("name", "unknown"),
            "count": int(row.get("count_count", 0)),
            "avg_ms": round(float(row.get("avg_latency", 0)), 1),
            "p50_ms": round(float(row.get("p50_latency", 0)), 1),
            "p95_ms": round(float(row.get("p95_latency", 0)), 1),
            "p99_ms": round(float(row.get("p99_latency", 0)), 1),
        }
        for row in result.data
    ]
    return {"window_hours": hours, "stages": stages}


@app.get("/admin/ttft-report")
def ttft_report(limit: int = QueryParam(default=20, ge=1, le=100)):
    """
    TTFT and TPOT computed from raw observation timestamps + token counts.
    Langfuse does not compute TPOT natively - this is the manual calculation.
    """
    observations = langfuse_client.api.observations.get_many(
        limit=limit, fields="core,basic,usage,time"
    )
    rows = []
    for obs in observations.data:
        if obs.type != "GENERATION":
            continue
        if not obs.completion_start_time:
            continue
        ttft_s = (obs.completion_start_time - obs.start_time).total_seconds()
        if obs.start_time is not None and obs.end_time is not None:
            e2e_s = (obs.end_time - obs.start_time).total_seconds()
        else:
            e2e_s = None

        output_tokens = (obs.usage_details or {}).get("output", 0)

        tpot_ms = None
        if output_tokens > 0 and obs.end_time is not None:
            generation_only_s = (
                obs.end_time - obs.completion_start_time
            ).total_seconds()
            tpot_ms = round((generation_only_s / output_tokens) * 1000, 1)
        if e2e_s is None:
            continue
        rows.append(
            {
                "observations_id": obs.id,
                "ttft_s": round(ttft_s, 2),
                "e2e_s": round(e2e_s, 2),
                "output_tokens": output_tokens,
                "tpot_ms_per_token": tpot_ms,
            }
        )
    if not rows:
        return {"message": "No streaming generations found yet."}

    return {
        "sample_size": len(rows),
        "avg_ttft_s": round(mean(r["ttft_s"] for r in rows), 2),
        "median_ttft_s": round(median(r["ttft_s"] for r in rows), 2),
        "avg_tpot_ms_per_token": round(
            mean(r["tpot_ms_per_token"] for r in rows if r["tpot_ms_per_token"]), 1
        ),
        "requests": rows,
    }


@app.get("/admin/drift-report")
async def drift_report(
    baseline_n: int = QueryParam(default=50, ge=5),
    current_n: int = QueryParam(default=50, ge=5),
):
    """
    Compares the earliest `baseline_n` logged requests against the most
    recent `current_n` - both data drift (query length) and output quality drift (citation rate) in one report.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        baseline_rows = await conn.fetch(
            "SELECT query_length, cites_source FROM request_log ORDER BY created_at ASC LIMIT $1",
            baseline_n,
        )
        current_rows = await conn.fetch(
            "SELECT query_length, cites_source FROM request_log ORDER BY created_at DESC LIMIT $1",
            current_n,
        )
    if len(baseline_rows) < 5 or len(current_rows) < 5:
        return {"error": "Not enough data yet - need at least 5 rows in each window."}
    baseline_lengths = [r["query_length"] for r in baseline_rows]
    current_lengths = [r["query_length"] for r in current_rows]

    length_z, length_p = two_sample_z_test_means(baseline_lengths, current_lengths)

    baseline_cites = sum(1 for r in baseline_rows if r["cites_source"])
    current_cites = sum(1 for r in current_rows if r["cites_source"])
    cite_z, cite_p = two_proportion_z_test(
        baseline_cites, len(baseline_rows), current_cites, len(current_rows)
    )

    return {
        "input_drift": {
            "baseline_mean_length": round(statistics.mean(baseline_lengths), 1),
            "current_mean_length": round(statistics.mean(current_lengths), 1),
            "z": round(length_z, 2),
            "p": round(length_p, 4),
            "flagged": length_p < 0.05,
        },
        "output_quality_drift": {
            "baseline_cite_rate": round(baseline_cites / len(baseline_rows), 3),
            "current_cite_rate": round(current_cites / len(current_rows), 3),
            "z": round(cite_z, 2),
            "p": round(cite_p, 4),
            "flagged": cite_p < 0.05,
        },
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
