from datetime import datetime, timedelta, timezone
import json

from dotenv import load_dotenv
import langfuse

load_dotenv()

from contextlib import asynccontextmanager
from fastapi import FastAPI, Query as QueryParam
from app.instrumentation import verify_langfuse_connection
from app.rag_pipeline import run_rag_pipeline

client = langfuse.get_client()


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
    result = client.api.legacy.metrics_v1.metrics(query=query)
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


@app.get("/health")
async def health():
    return {"status": "ok"}
