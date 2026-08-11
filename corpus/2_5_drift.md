
**Pillar 2 — Observability | Tier 1 | Run Locally**

> **Where you are:** 2.4 built a heuristic scorer (`cites_source`, `reasonable_length`) and used it once, to compare two prompt variants at a single point in time. Today that same scorer becomes a _continuous_ signal — wired into live traffic, logged over time, and watched for silent degradation that no exception, no error status, and no single bad trace would ever reveal on its own.

---

## Session Goals

A RAG pipeline can degrade in three structurally different ways, and each one is invisible to the tools you've built so far. 2.1's tracing catches a single slow or broken request. 2.2's cost tracking catches a single expensive one. Neither catches a _gradual_ shift — the citation rate quietly dropping from 82% to 40% over a week because someone reverted a prompt label, or query patterns slowly shifting until your mock retrieval's fixed context stops being relevant to what people are actually asking. Today you build the detector for that second, harder class of problem.

**The deliverable:** Every live request now gets scored and logged to Postgres. A `/admin/drift-report` endpoint compares a baseline window against a current window using statistical tests — reusing and extending the exact z-test machinery from 2.4 — and flags real drift you'll deliberately inject to prove the detector actually works before trusting it on real unknown data.

---

## The 3 Concepts You Must Own After This Session

1. **Data drift vs concept drift vs output quality drift** — three genuinely different failure modes with different causes, different detection signals, and different fixes. Confusing them means building the wrong monitor.
2. **Silent degradation signals** — why "the API still returns 200 OK, the JSON still looks fine" is exactly the situation drift detection exists for, and why a single trace, however carefully inspected, can never reveal it — only a signal tracked _across_ traces, over time, can.
3. **Statistical significance cuts both ways** — 2.4 taught you that small samples can miss real effects. Today's mirror-image lesson: with enough samples, even a trivial, meaningless difference becomes "statistically significant." A drift detector needs both a significance test _and_ a minimum effect-size threshold, or it cries wolf constantly at scale.

---

## Why "It's Still Returning 200 OK" Is Not Reassurance

### The MERN Instinct

In Express, if a route handler throws, you get a 500, a stack trace, an alert. If it doesn't throw, the assumption is usually "it worked." That assumption holds for CRUD — either the database write succeeded or it didn't, and there's rarely a silent middle ground.

LLM pipelines break that assumption completely. A request can execute every step successfully — retrieval returns chunks, the model generates a plausible-sounding paragraph, the endpoint returns `200` with well-formed JSON — and still be _wrong_, or worse than it used to be, with nothing in the response shape signaling it. This is the same "silent degradation" theme from 2.1's framing of why tracing matters at all, but today it's about degradation that unfolds gradually across many requests rather than failing obviously in one.

```
A single bad request:                Drift over time:
┌─────────────────────┐              ┌──────────────────────────────────┐
│ trace looks weird    │              │ day 1: cites_source rate ≈ 82%   │
│ → you'd catch it      │              │ day 3: cites_source rate ≈ 71%   │
│   inspecting ONE      │              │ day 5: cites_source rate ≈ 58%   │
│   trace in the UI     │              │ day 7: cites_source rate ≈ 40%   │
│   (2.1's job)         │              │                                  │
│                        │              │ Every SINGLE request in this    │
│                        │              │ window still returns 200 OK.    │
│                        │              │ No trace, viewed alone, looks    │
│                        │              │ obviously broken. Only the      │
│                        │              │ TREND reveals the problem.      │
│                        │              │ (today's job)                   │
└─────────────────────┘              └──────────────────────────────────┘
```

---

## Three Kinds of Drift — Precisely Defined, With RAG-Specific Examples

### Data Drift — the inputs change

The distribution of what people are _asking_ shifts from what your system was built and tested against. The model and pipeline are unchanged; the traffic hitting them looks different.

**RAG example:** your portfolio API was tested against short, single-topic questions ("What is FastAPI?"). Real usage starts sending long, multi-part questions ("Compare FastAPI and Flask for a team migrating from Express, considering async support, ecosystem maturity, and our Postgres setup"). Same model, same retrieval, same prompt — but your system was never validated against this input shape, and it may perform worse on it without anyone having changed anything on your end.

### Concept Drift — the correct answer changes

The _relationship_ between a given input and its correct output shifts over time, independent of whether the input distribution changed at all.

**RAG example:** a retrieved document says "FastAPI's latest stable release is 0.100." That was true when the document was written. Six months later, someone asks the exact same question, worded identically — but the correct answer has changed, because the world changed, not the question. Your citation-rate monitor won't catch this: the model is faithfully citing a document that is now _wrong_. This is squarely a document-freshness problem, not a prompt or model problem — which is exactly why it's Pillar 10.5's job ("embedding pipeline operations... index drift"), not something a query-length or citation-rate signal can detect. Flagging it here so you can name it correctly in an interview; the hands-on fix requires real document indexing, which your mock 3-chunk retrieval can't meaningfully simulate.

### Output Quality Drift — the system's behavior changes

Input distribution is stable. The real-world "correct answer" hasn't changed. But the system's _output_ quality degrades anyway — because something in the pipeline itself silently changed.

**RAG example, and today's primary lab:** someone reverts the `production` label on your `rag-answer` prompt from v2 (citation instruction) back to v1 (baseline) — maybe by accident, maybe someone testing something and forgetting to revert. No code changed. No deploy happened. The API still returns `200` with plausible-looking answers. But your citation rate silently craters, because the pipeline itself is no longer doing what it was doing yesterday.

---

## Part A: Wire Live Scoring Into Every Request, Log to Postgres

2.4's `score_answer()` only ran inside the offline A/B test script. Drift detection needs a _continuous_ signal from live traffic, so today it moves into the `/query` endpoint itself — and gets logged somewhere queryable over time.

**Why Postgres, not another Langfuse query:** you already have it from Pre-Work, you already understand its query model, and — unlike guessing at whether Langfuse's metadata fields are reliably queryable by custom keys through their Metrics/Observations APIs — a table you control has a schema you know cold. Not every observability signal needs to live in your LLM-ops vendor. Knowing when a simple owned table beats reaching for the vendor API again is itself a maturity signal.

### Add the Table

If Pre-Work's P.2 already set up a shared `asyncpg` connection pool, reuse that pattern instead of duplicating it — the following is self-contained in case you need it fresh:

```python
# app/db.py
import asyncpg
import os

_pool: asyncpg.Pool | None = None


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(dsn=os.environ["DATABASE_URL"], min_size=1, max_size=5)
    return _pool


async def init_request_log_table():
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS request_log (
                id SERIAL PRIMARY KEY,
                trace_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                query TEXT NOT NULL,
                query_length INT NOT NULL,
                answer_length INT,
                cites_source BOOLEAN,
                reasonable_length BOOLEAN,
                prompt_version TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)
```

**Adjust `DATABASE_URL`** to whatever env var name your P.2 Postgres setup actually used if it differs.

### Extend the Existing Lifespan (Don't Create a Second One)

You already have a `lifespan` function in `app/main.py` from 2.1. Extend it — don't add a competing one:

```python
from app.db import init_request_log_table, get_pool

@asynccontextmanager
async def lifespan(app: FastAPI):
    verify_langfuse_connection()
    await init_request_log_table()
    yield
```

### Wire Scoring + Logging Into `/query`

```python
@app.get("/query")
async def query_endpoint(
    q: str = QueryParam(..., min_length=3),
    user_id: str = QueryParam(default="anonymous"),
    prompt_version: int | None = QueryParam(default=None),
):
    result = run_rag_pipeline(query=q, user_id=user_id, prompt_version=prompt_version)
    scores = score_answer(result["trace_id"], result["answer"])  # from 2.4 — now runs on every live request

    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO request_log
                (trace_id, user_id, query, query_length, answer_length,
                 cites_source, reasonable_length, prompt_version)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
            result["trace_id"], user_id, q, len(q), len(result["answer"]),
            scores["cites_source"], scores["reasonable_length"],
            str(prompt_version) if prompt_version else "production",
        )

    return result
```

**Worth knowing, not building today:** this only works because `score_answer()` is cheap regex checks — microseconds of added latency. If Pillar 11 later swaps in an LLM-as-judge scorer, running that _inline_ on every live request would add seconds of latency per request. That's exactly why production eval pipelines usually score asynchronously, offline, in a batch job — not inline in the request path. Heuristic scores can afford to be live; LLM-judge scores generally can't.

---

## Part B: Detect Data Drift — Query Length

### The Statistical Test — Extending 2.4's Z-Test to Continuous Data

2.4 built a two-_proportion_ z-test (comparing two rates, like citation rate). Query length isn't a proportion — it's a continuous measurement. The right tool is a two-_sample_ z-test for a difference in means, same underlying approach (compare an observed difference against its expected variability under the null hypothesis of no real difference), same stdlib implementation technique (`math.erf` for the normal CDF, no new dependency):

```python
import math
import statistics

def two_sample_z_test_means(sample_a: list[float], sample_b: list[float]) -> tuple[float, float]:
    """
    Two-sample z-test for a difference in means. Appropriate here because
    both windows have 30+ samples — large enough for the sampling
    distribution of the mean to be approximately normal (Central Limit
    Theorem), which is what justifies using a z-test instead of a t-test.
    For much smaller windows, a t-test would be the more defensible choice.
    """
    mean_a, mean_b = statistics.mean(sample_a), statistics.mean(sample_b)
    var_a, var_b = statistics.variance(sample_a), statistics.variance(sample_b)
    n_a, n_b = len(sample_a), len(sample_b)

    se = math.sqrt(var_a / n_a + var_b / n_b)
    if se == 0:
        return 0.0, 1.0
    z = (mean_a - mean_b) / se
    cdf = 0.5 * (1 + math.erf(abs(z) / math.sqrt(2)))
    p_value = 2 * (1 - cdf)
    return z, p_value
```

### Add `/admin/drift-report`

```python
@app.get("/admin/drift-report")
async def drift_report(
    baseline_n: int = QueryParam(default=50, ge=5),
    current_n: int = QueryParam(default=50, ge=5),
):
    """
    Compares the earliest `baseline_n` logged requests against the most
    recent `current_n` — both data drift (query length) and output quality
    drift (citation rate) in one report.
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
        return {"error": "Not enough data yet — need at least 5 rows in each window."}

    baseline_lengths = [r["query_length"] for r in baseline_rows]
    current_lengths = [r["query_length"] for r in current_rows]
    length_z, length_p = two_sample_z_test_means(baseline_lengths, current_lengths)

    baseline_cites = sum(1 for r in baseline_rows if r["cites_source"])
    current_cites = sum(1 for r in current_rows if r["cites_source"])
    cite_z, cite_p = two_proportion_z_test(  # from 2.4, unchanged
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
```

### Simulate Data Drift and Verify the Detector Catches It

Send a baseline of short, uniform questions:

```bash
for q in "What is FastAPI" "What does Pydantic do" "How does Docker work" \
         "What is Redis" "What is a vector database" "What is RAG"; do
  for i in 1 2 3 4 5 6 7 8; do
    curl -s "http://localhost:8000/query?q=$(python3 -c "import urllib.parse;print(urllib.parse.quote('$q'))")&user_id=baseline" > /dev/null
  done
done
```

Check the report — baseline vs "current" (which right now is the same data) should show `input_drift.flagged: false`, roughly equal means.

Now inject deliberately drifted input — much longer, multi-clause questions:

```bash
curl -s "http://localhost:8000/query?q=$(python3 -c "import urllib.parse;print(urllib.parse.quote('Considering a team migrating from Express and MongoDB to FastAPI and Postgres, what are the tradeoffs around dependency injection, async database drivers, and container memory limits on constrained hardware'))")&user_id=drift-test" > /dev/null
# Repeat with 10-15 similarly long, multi-part questions
```

Re-check `/admin/drift-report?baseline_n=48&current_n=15` — you should now see `input_drift.flagged: true`, with `current_mean_length` substantially higher than `baseline_mean_length`. **This is the point of simulating known drift before trusting the detector on real traffic** — you're verifying the detector actually fires on a real, injected shift before relying on it to catch one you didn't cause on purpose.

---

## Part C: Detect Output Quality Drift — Simulate a Silent Regression

This is the scenario from the concept section made concrete: revert the `production` label without changing any code, and confirm the drift detector catches it even though nothing throws, nothing 500s, and every individual response still looks like a normal answer.

```python
# Simulate someone accidentally reverting production — run once
from langfuse import get_client
langfuse = get_client()

langfuse.update_prompt(name="rag-answer", version=1, new_labels=["production"])
print("Production reverted to v1 (no citation instruction) — simulating a regression")
```

Send a batch of normal-looking requests against this "regressed" state:

```bash
for i in $(seq 1 15); do
  curl -s "http://localhost:8000/query?q=What+does+concept+$i+mean&user_id=regression-test" > /dev/null
done
```

Check the report:

```bash
curl "http://localhost:8000/admin/drift-report?baseline_n=50&current_n=15" | python -m json.tool
```

Expect `output_quality_drift.flagged: true` — `current_cite_rate` should have collapsed toward near-zero, `baseline_cite_rate` still reflecting the healthy v2 period. Every one of those 15 "regressed" requests returned `200 OK` with a normal-looking answer. Nothing about any single response would have told you something was wrong. Only the trend does.

**Restore production before moving on** — don't leave your app in the regressed state:

```python
langfuse.update_prompt(name="rag-answer", version=2, new_labels=["production"])
print("Production restored to v2")
```

---

## Common Failure Modes

### Failure 1: Drift report flags something trivial as "significant"

**Symptom:** `baseline_mean_length: 42.1`, `current_mean_length: 43.8`, `flagged: true` — a 1.7-character difference somehow counts as significant.

**Root Cause:** you ran this with very large `baseline_n`/`current_n` values. At large sample sizes, a z-test's statistical power increases enough that even a practically meaningless difference clears `p < 0.05`. This is the mirror image of 2.4's small-sample lesson: small samples miss real effects; large samples flag trivial ones. **Fix:** a production drift monitor should combine the significance test with a _minimum effect-size threshold_ — e.g., only flag input drift if `p < 0.05` **and** `abs(current_mean_length - baseline_mean_length) > 15` characters, tuned to what's actually operationally meaningful for your system, not just whatever the test happens to detect.

### Failure 2: Output quality drift never flags, even after the simulated regression

**Diagnose:**

1. Confirm the `update_prompt` call actually succeeded — check the Langfuse UI's Prompts page to see which version currently carries the `production` label.
2. Confirm you sent _new_ requests after reverting — rows already in `request_log` from before the revert don't change retroactively.
3. Confirm `current_n` in your report query is small enough to actually only capture the post-revert requests — if `current_n=50` but you only sent 15 regressed requests, the "current" window is a mix of 35 healthy
    - 15 regressed rows, diluting the signal. Match `current_n` to roughly how many requests you actually sent in the drifted condition.

### Failure 3: Forgot to restore `production` to v2 after the simulation

**Symptom:** later sessions' traffic quietly stays un-cited, and you don't notice for a while — which is, pointedly, the exact failure mode this session is about. Always restore state deliberately after a simulated regression, and consider it good practice to re-check `langfuse.get_prompt("rag-answer", label="production").version` after any experiment that touches labels, rather than assuming it reverted correctly.

---

## Verification Checklist

- [x] `request_log` table created; existing `lifespan` extended, not duplicated
- [x] `/query` scores and logs every live request, not just offline experiments
- [x] `/admin/drift-report` returns both `input_drift` and `output_quality_drift` sections
- [x] Sent a uniform baseline batch; report shows `input_drift.flagged: false` against itself
- [x] Sent deliberately long/complex queries; report correctly flags `input_drift.flagged: true`
- [x] Simulated the production-label revert; report correctly flags `output_quality_drift.flagged: true`
- [x] Restored `production` to v2 afterward and verified it in the Langfuse UI
- [x] You can state, without notes, the difference between data drift, concept drift, and output quality drift, with a RAG-specific example of each
- [x] You can explain why a citation-rate monitor would never catch concept drift, specifically

---

## Directory Structure After This Session

```
ai-infra-portfolio/
├── app/
│   ├── main.py               ← lifespan extended; /query now scores+logs; /admin/drift-report added
│   ├── db.py                  ← NEW — asyncpg pool + request_log table
│   ├── instrumentation.py    ← unchanged
│   ├── llm_client.py         ← unchanged
│   └── rag_pipeline.py       ← unchanged from 2.4
├── scripts/                   ← unchanged from 2.4
├── .env                       ← unchanged (DATABASE_URL should already exist from Pre-Work)
└── requirements.txt            ← unchanged — no new packages
```

---

## Evening Integration — Before You Close the Laptop

```bash
git add app/main.py app/db.py
git commit -m "[pillar-2.5] live scoring + Postgres request_log; drift detector for data and output quality drift"
git push
```

**3-sentence journal:**

1. What you built today and what it does
2. What broke and the exact error / symptom
3. What you'd do differently if starting over

---

## Commit Message

```
[pillar-2.5] live scoring + Postgres request_log; drift detector for data and output quality drift
```

---

## Preview: What 2.6 Builds Directly On This

**Session 2.6 (Golden Test Sets & Regression Evals in CI)** takes today's manual "send drifted traffic, check the report" workflow and makes it automatic, running on every pull request instead of on-demand from a terminal. The `request_log` table and z-test machinery you built today don't disappear — but 2.6 adds a _fixed, curated_ set of test cases (a precursor to Pillar 11's full golden set) that runs in GitHub Actions and fails a PR outright if a change measurably regresses quality, rather than waiting for you to notice a slow drift in production days later. Today you built the ability to detect drift after it's already happened; 2.6 starts building the ability to block it before it ships.

---

## Quick Reference Card

```
THREE KINDS OF DRIFT:
  Data drift          = inputs change (query shape/distribution shifts)
  Concept drift        = the correct answer changes (world changed, docs stale)
                          — NOT caught by query-length or citation monitors;
                            needs document freshness tracking (Pillar 10.5)
  Output quality drift = system behavior changes silently (prompt reverted,
                          model swapped, pipeline regression) — inputs and
                          ground truth both stable, output quality isn't

WHY THIS MATTERS:
  Every degraded request in a drift scenario still returns 200 OK.
  No single trace looks obviously broken. Only a TREND across many
  requests reveals it.

STATISTICAL TESTS (both stdlib-only, math.erf for normal CDF):
  two_proportion_z_test()      → from 2.4, for rates (citation rate, etc.)
  two_sample_z_test_means()    → new today, for continuous values (length, etc.)

THE TWO-DIRECTIONAL STATS TRAP:
  Small samples (2.4)  → real effects can go undetected (p >= 0.05 ≠ "no effect")
  Large samples (2.5)  → trivial effects become "significant"
  Fix: combine a significance test with a MINIMUM EFFECT SIZE threshold

DESIGN PRINCIPLE:
  Heuristic scores (cheap, regex-based) → safe to run inline, on every live request
  LLM-judge scores (expensive, slow)     → run offline/async, not in the request path
```