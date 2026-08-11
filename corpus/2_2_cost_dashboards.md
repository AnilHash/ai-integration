
**Pillar 2 — Observability | Tier 1 | Local**
**Prerequisite:** Session 2.1 (Langfuse tracing wired into your RAG endpoint)

---

## 1. Session Goal

Every LLM call your API makes has a real dollar cost, even when the model behind it is free. This session makes that cost visible per request, per user, and per feature — token counts in, cost out, stored somewhere you can query and alert on. By the end, every call to your endpoint writes a row you can aggregate into a cost report, and a script exists that would page someone if spend crossed a threshold.

This is the session that turns "I called an LLM" into "I know exactly what that call cost, who triggered it, and whether we're on budget" — the difference between a script and production infrastructure.

---

## 2. MERN Bridge

You've built this pattern before, just for a different resource.

| MERN world | This session |
|---|---|
| Stripe metering: tag every API call with a customer ID to bill usage | Tag every LLM call with `user_id` + `feature` to attribute cost |
| Express rate-limiter middleware (`express-rate-limit`) checking requests/min | Budget script checking dollars/day |
| Logging middleware writing `{method, path, statusCode, duration}` to Mongo | Cost wrapper writing `{model, tokens, cost_usd, latency_ms}` to Postgres |
| A MongoDB aggregation pipeline (`$group by userId`) for a usage report | A Postgres `GROUP BY` query for a cost report |

Same instinct as request logging middleware in Express — you're just billing tokens instead of counting hits.

---

## 3. Three Core Concepts

### 3.1 Token Counting

Every chat completion response carries a `usage` object: `prompt_tokens`, `completion_tokens`, `total_tokens`. You don't estimate this — the model server counts it exactly and hands it back. Ollama's OpenAI-compatible endpoint (`/v1/chat/completions`) returns this natively for non-streaming calls, aggregated internally from its own `prompt_eval_count` and `eval_count` fields. You never need a separate tokenizer library for this lab — the number comes from the response, not from counting characters yourself.

```
Request  →  Ollama /v1/chat/completions  →  response.usage
                                              {
                                                "prompt_tokens": 142,
                                                "completion_tokens": 58,
                                                "total_tokens": 200
                                              }
```

**One gotcha to file away for Pillar 3.4 (SSE streaming):** the `usage` field is only reliably populated on non-streaming responses by default. Streaming responses need `"stream_options": {"include_usage": true}` in the request, and even then usage only shows up in the *final* chunk with an empty `choices` array. Today's lab is non-streaming, so this doesn't bite you yet — but remember it when you build the streaming endpoint later, or your token counts will silently come back `None`.

### 3.2 Cost Attribution Per User / Feature

Token counts alone aren't a cost dashboard — you need to know *whose* tokens and *which feature* burned them. This means every LLM call needs two tags attached before it ever hits the model: a `user_id` and a `feature` name (e.g. `"rag-query"`, `"summarize"`, `"eval-judge"`). Attribution is a schema decision made at write time, not something you can reconstruct later from a pile of untagged logs.

phi3:mini running locally costs you nothing — no API bill exists to attribute. So this lab applies a **synthetic pricing table**: you price every local call as if it were hitting a real paid API (gpt-4o-mini's published rate: $0.15 / 1M input tokens, $0.60 / 1M output tokens, as of mid-2026). This isn't a fudge — it's the honest way to build and test a cost pipeline for free. The pipeline — counting, tagging, storing, aggregating, alerting — is identical whether the number multiplying your tokens is a real invoice line or a stand-in rate. When Pillar 12 puts LiteLLM in front of a real paid provider, you swap the pricing table for real rates and change nothing else.

### 3.3 Budget Alerting

A cost dashboard nobody looks at is worthless. Budget alerting closes the loop: a scheduled check compares accumulated spend against a threshold and raises a signal when it's crossed. At this stage — no Prefect or Airflow yet, that's Pillar 3 — this is a standalone script you'd run on a cron schedule. The design question that matters isn't the cron syntax, it's *what counts as "spend" and over what window* (today so far? trailing 24h? per-user or aggregate?). Get the window wrong and you either alert too late or drown yourself in false positives — same failure mode as a badly-tuned Express rate limiter.

---

## 4. Architecture — Where This Sits in the Request Path

```mermaid
flowchart TD
    A[Client request] --> B[FastAPI /query endpoint]
    B --> C[Langfuse generation observation opens]
    C --> D[Ollama /v1/chat/completions — phi3:mini]
    D --> E[response.usage: prompt_tokens, completion_tokens]
    E --> F[pricing table: calculate_cost_usd]
    F --> G[Langfuse generation.update — usage_details, cost_details]
    F --> H[Postgres INSERT — request_logs]
    G --> I[Langfuse Cloud dashboard]
    H --> J[FastAPI /dashboard/costs/summary]
    H --> K[scripts/check_budget.py — cron]
```

Two destinations for the same numbers, on purpose:

- **Langfuse** gets `usage_details` + `cost_details` attached to the generation you already opened in 2.1. This is what makes Langfuse's built-in Users/cost views work, and it's free — you're just filling in fields on a span you already have open.
- **Postgres** gets its own row in a `request_logs` table. This is the part that's actually yours: you own the schema, you own the aggregation queries, and it doesn't disappear if you ever change observability vendors. In an interview, "I built the cost dashboard from my own Postgres table, not just Langfuse's UI" is the stronger answer.

---

## 5. Environment Verification — Do This Before Writing Any Code

`load_dotenv()` must run before anything constructs the Langfuse client or the Postgres pool — both read env vars once, at first use, and cache the result. If FastAPI/uvicorn imports a router that builds either client before `load_dotenv()` has run, you get a silently broken client with no error.

**`.env` additions for this session** (append to what 2.1 already has):

```
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
DATABASE_URL=postgresql://user:pass@localhost:5432/portfolio
```

**Import order in `app/main.py` — `load_dotenv()` first, before any other project import:**

```python
# app/main.py
from dotenv import load_dotenv
load_dotenv()                      # must run before the next two lines

from fastapi import FastAPI
from app.routers import query, dashboard
from app.observability.langfuse_client import verify_langfuse_auth

app = FastAPI()
app.include_router(query.router)
app.include_router(dashboard.router)

@app.on_event("startup")
async def check_observability():
    if not verify_langfuse_auth():
        raise RuntimeError(
            "Langfuse auth_check() failed — traces will not be recorded. "
            "Check LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL."
        )
```

**Standalone diagnostic — run this before touching the app, confirm it prints clean:**

```python
# scripts/env_check.py — run: python scripts/env_check.py
from dotenv import load_dotenv
load_dotenv()
import os

required = ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL", "DATABASE_URL"]
missing = [k for k in required if not os.getenv(k)]

print("MISSING:", missing) if missing else print("All required env vars loaded.")
```

---

## 6. Lab Build

### 6.1 Install

```bash
pip install "langfuse>=4.0,<5.0" openai asyncpg python-dotenv --break-system-packages
```

Langfuse's Python SDK was rewritten in v4 (released March 2026) — the client API is `get_client()` + `start_as_current_observation(...)`, not the older `Langfuse().trace()` pattern you may see in older tutorials. If you find a blog post using `langfuse.trace(name=...)`, it's targeting the legacy v2/v3 SDK — don't copy it.

### 6.2 Postgres schema

```sql
-- migrations/002_request_logs.sql
CREATE TABLE IF NOT EXISTS request_logs (
    id                BIGSERIAL PRIMARY KEY,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    user_id           TEXT NOT NULL,
    feature           TEXT NOT NULL,
    model             TEXT NOT NULL,
    prompt_tokens     INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    cost_usd          NUMERIC(12, 8) NOT NULL,
    latency_ms        NUMERIC(10, 1) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_request_logs_created_at ON request_logs (created_at);
CREATE INDEX IF NOT EXISTS idx_request_logs_user_id    ON request_logs (user_id);
CREATE INDEX IF NOT EXISTS idx_request_logs_feature    ON request_logs (feature);
```

Run it against your existing Dockerized Postgres (from P.4), don't spin up a second instance.

### 6.3 Synthetic pricing table

```python
# app/costs/pricing.py
"""
Illustrative $/token rates — phi3:mini costs nothing locally, but this
table prices every call as if it hit a real paid API, so the cost
pipeline behaves exactly like it would in production.

Source: OpenAI published gpt-4o-mini pricing, confirmed July 2026:
$0.15 / 1M input tokens, $0.60 / 1M output tokens.
Rates change — re-check before trusting this in Pillar 4.6's cost model doc.
"""

MODEL_PRICING_USD_PER_MILLION = {
    "phi3:mini":     {"input": 0.15, "output": 0.60},
    "gemma2:2b":     {"input": 0.15, "output": 0.60},
    "qwen2.5:1.5b":  {"input": 0.15, "output": 0.60},
}


def calculate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    rates = MODEL_PRICING_USD_PER_MILLION.get(model)
    if rates is None:
        raise ValueError(f"No pricing entry for model '{model}' — add one before shipping this call.")
    input_cost = (prompt_tokens / 1_000_000) * rates["input"]
    output_cost = (completion_tokens / 1_000_000) * rates["output"]
    return round(input_cost + output_cost, 8)
```

### 6.4 Langfuse client singleton

```python
# app/observability/langfuse_client.py
from langfuse import get_client

langfuse = get_client()


def verify_langfuse_auth() -> bool:
    """auth_check() returns False on bad credentials — it does NOT raise.
    A try/except around it catches nothing. Call this at startup and
    fail loudly, or you get an app that runs fine while every trace
    silently vanishes."""
    return langfuse.auth_check()
```

### 6.5 The cost-tracking wrapper — the core of this session

This wraps the Ollama call you already built in 2.1, adding token extraction, cost calculation, Langfuse logging, and a Postgres write around it.

```python
# app/services/llm_cost_tracking.py
import time
from openai import OpenAI

from app.observability.langfuse_client import langfuse
from app.costs.pricing import calculate_cost_usd
from app.db.repository import insert_request_log

ollama_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")


async def call_model_with_cost_tracking(
    *, prompt: str, model: str, user_id: str, feature: str
) -> dict:
    start = time.perf_counter()

    with langfuse.start_as_current_observation(
        as_type="generation",
        name=f"{feature}-completion",
        model=model,
        input=[{"role": "user", "content": prompt}],
    ) as generation:

        response = ollama_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )

        usage = response.usage
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        cost_usd = calculate_cost_usd(model, prompt_tokens, completion_tokens)
        latency_ms = round((time.perf_counter() - start) * 1000, 1)

        generation.update(
            output=response.choices[0].message.content,
            usage_details={"input": prompt_tokens, "output": completion_tokens},
            cost_details={"total": cost_usd},
            metadata={"user_id": user_id, "feature": feature},
        )

    await insert_request_log(
        user_id=user_id,
        feature=feature,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
    )

    return {
        "content": response.choices[0].message.content,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost_usd": cost_usd,
        "latency_ms": latency_ms,
    }
```

Note why Langfuse can't auto-price this for you: Langfuse only infers cost automatically for known providers (OpenAI, Anthropic out of the box). `phi3:mini` is an unrecognized model name to it — so `cost_details` has to be computed and passed explicitly. That's not a workaround, that's the normal pattern for any self-hosted model.

### 6.6 Repository layer

```python
# app/db/repository.py
from app.db.pool import get_pool


async def insert_request_log(
    *, user_id, feature, model, prompt_tokens, completion_tokens, cost_usd, latency_ms
):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO request_logs
                (user_id, feature, model, prompt_tokens, completion_tokens, cost_usd, latency_ms)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            user_id, feature, model, prompt_tokens, completion_tokens, cost_usd, latency_ms,
        )
```

### 6.7 Wire it into the endpoint

```python
# app/routers/query.py
from fastapi import APIRouter
from pydantic import BaseModel
from app.services.llm_cost_tracking import call_model_with_cost_tracking

router = APIRouter()


class QueryRequest(BaseModel):
    prompt: str
    user_id: str
    feature: str = "rag-query"


@router.post("/query")
async def query(req: QueryRequest):
    return await call_model_with_cost_tracking(
        prompt=req.prompt,
        model="phi3:mini",
        user_id=req.user_id,
        feature=req.feature,
    )
```

### 6.8 The dashboard endpoint

A "cost dashboard" at this stage is a reporting endpoint, not a frontend — frontend isn't the skill this pillar tests.

```python
# app/routers/dashboard.py
from fastapi import APIRouter, Query
from app.db.pool import get_pool

router = APIRouter(prefix="/dashboard", tags=["dashboard"])

# Whitelist, not user-supplied SQL — never interpolate raw query params
# into a query string. This is the one place MongoDB's document model
# hides a whole injection class that SQL doesn't.
_GROUP_COLUMNS = {
    "day": "date_trunc('day', created_at)",
    "user": "user_id",
    "feature": "feature",
}


@router.get("/costs/summary")
async def cost_summary(group_by: str = Query("day", pattern="^(day|user|feature)$")):
    column = _GROUP_COLUMNS[group_by]
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT {column} AS bucket,
                   COUNT(*) AS requests,
                   SUM(prompt_tokens + completion_tokens) AS total_tokens,
                   SUM(cost_usd) AS total_cost_usd,
                   AVG(latency_ms) AS avg_latency_ms
            FROM request_logs
            GROUP BY bucket
            ORDER BY bucket DESC
            """
        )
    return [dict(row) for row in rows]
```

Verify: `curl "http://localhost:8000/dashboard/costs/summary?group_by=feature"` after a handful of `/query` calls should return one row per feature with aggregated cost and token counts.

### 6.9 Budget alerting

```python
# scripts/check_budget.py — run manually, or via cron: */30 * * * *
import asyncio
from app.db.pool import get_pool

DAILY_BUDGET_USD = 5.00  # arbitrary ceiling for the portfolio project


async def check_budget():
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT COALESCE(SUM(cost_usd), 0) AS spent
            FROM request_logs
            WHERE created_at >= date_trunc('day', now())
            """
        )
    spent = float(row["spent"])
    if spent >= DAILY_BUDGET_USD:
        print(f"ALERT: daily cost ${spent:.4f} has crossed ${DAILY_BUDGET_USD:.2f} budget")
    else:
        print(f"OK: ${spent:.4f} / ${DAILY_BUDGET_USD:.2f} spent today")


if __name__ == "__main__":
    asyncio.run(check_budget())
```

Real orchestration (Prefect scheduling this instead of cron) comes in Pillar 3. Cron is the honest answer for where you are right now — don't reach for Prefect early just because it's coming later.

---

## 7. Verify the Lab Worked

Run these in order:

```bash
python scripts/env_check.py
# → All required env vars loaded.

uvicorn app.main:app --reload
# → startup should NOT raise the Langfuse auth RuntimeError

curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is PagedAttention?", "user_id": "test-user", "feature": "rag-query"}'
# → JSON response including prompt_tokens, completion_tokens, cost_usd, latency_ms

curl "http://localhost:8000/dashboard/costs/summary?group_by=feature"
# → one row for "rag-query" with the aggregated numbers

python scripts/check_budget.py
# → OK: $0.0000X / $5.00 spent today
```

Then open Langfuse Cloud → your trace from the `/query` call → the generation should show `usage_details` and `cost_details` populated, not blank.

---

## 8. Common Failure

**Symptom:** the app runs fine, `/query` returns real responses, Postgres rows appear correctly — but Langfuse Cloud shows nothing, or shows traces with no cost/usage attached.

**Cause:** `get_client()` is a singleton that reads `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_BASE_URL` the first time it's called. If any module imports `app.observability.langfuse_client` before `load_dotenv()` has run in `app/main.py`, the client initializes with empty credentials — and `auth_check()` returns `False` rather than raising, so nothing crashes. You get a fully functional app that quietly never talks to Langfuse.

**Diagnose it:**
1. Run `scripts/env_check.py` standalone first — confirms the vars exist in `.env` at all.
2. Add the `verify_langfuse_auth()` startup check shown in §5 — this converts a silent failure into a loud one at boot instead of a mystery three days later when you check the Langfuse UI and it's empty.
3. If startup still passes but traces don't appear: check the *import order* in `main.py`. `load_dotenv()` must be the literal first executed line, before the `from app.routers import ...` line — Python evaluates imports top to bottom, and any router that imports the Langfuse client transitively triggers `get_client()` before your `load_dotenv()` call if the import order is wrong.

This is exactly the class of bug the "silently-wrong health check is worse than no health check" rule exists for — a `try/except` around `auth_check()` catches nothing, because there's nothing to catch.

---

## 9. Interview Tie-In

**"How do you track LLM costs in production?"** — junior answer stops at "I count tokens and multiply by price." Senior answer: tokens come straight off the provider response, not a local tokenizer guess; cost is attributed at write time by user and feature, not reconstructed later; and the numbers land in two places — the observability vendor (for humans debugging one trace) and your own database (for aggregation and alerting you don't want to depend on a vendor UI for). Close with: *"and here's the GitHub where the request_logs table and the dashboard endpoint are."*

If asked why self-hosted models need a manual pricing table: because there's no invoice to read cost off of — the whole point of a synthetic pricing table is proving the pipeline works before it ever touches a real bill.

---

## 10. Commit

```
[pillar-2.2] token counting, cost attribution, and budget alerting on the RAG endpoint
```

---

## What's Next

**2.3 — Latency Profiling.** You already have `latency_ms` sitting in `request_logs` from this session — 2.3 breaks that single number into TTFT vs TPOT vs end-to-end, and adds timing spans to every stage of the RAG pipeline (not just the LLM call) to find out where time actually goes. Don't start it until this session's four verification checks in §7 all pass clean.
