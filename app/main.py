from datetime import datetime, timedelta, timezone
import json
from statistics import mean, median

from dotenv import load_dotenv
import langfuse

load_dotenv()

from contextlib import asynccontextmanager
from fastapi import FastAPI, Query as QueryParam
from app.instrumentation import verify_langfuse_connection
from app.rag_pipeline import run_rag_pipeline

langfuse_client = langfuse.get_client()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not verify_langfuse_connection():
        raise RuntimeError(
            "Langfuse auth_check() failed - traces will not be recorded. "
            ""
            "Check env variables."
        )
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/query")
async def query_endpoint(
    q: str = QueryParam(..., min_length=3, description="User query string"),
    user_id: str = QueryParam(
        default="anonymous", description="User identifier for cost attribution"
    ),
):
    result = run_rag_pipeline(query=q, user_id=user_id)
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


@app.get("/health")
async def health():
    return {"status": "ok"}
