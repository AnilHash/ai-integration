**Pillar 2 — Observability | Tier 1 | Run Locally**

> **Where you are:** This is the capstone of Pillar 2. 2.2 gave you cost. 2.3 gave you latency. 2.5 gave you drift. 2.6 gave you a pre-merge gate. Today those four signals become one coherent alert matrix — with real severity tiers, a real answer to "what deserves a 3am page," and one genuinely important infrastructure lesson: not everything you'd want to automate is actually automatable when your whole system runs on a laptop with no deployed target.

---

## Session Goals

Four disconnected signals aren't a monitoring system — they're four things you'd have to remember to check manually. Today you assemble them into a matrix: for each signal, decide whether it's threshold-based or anomaly-based, what severity it deserves, where it should run, and whether it should interrupt a human at 3am or just sit in a dashboard until tomorrow. You'll also hit a real constraint worth understanding clearly: some of this can run in Langfuse's cloud regardless of whether your laptop is even on; some of it fundamentally can't be automated in the cloud at all, because it depends on a local Postgres table that only exists on your machine.

**The deliverable:** `docs/alerting.md` — the alert matrix as a real design artifact in your repo. Three Langfuse-native Monitors covering the cloud-side threshold signals. A local script plus a Windows Task Scheduler entry covering the signals that can only run against your local Postgres data. And error-rate tracking, which didn't exist before today.

---

## The 3 Concepts You Must Own After This Session

1. **Threshold vs anomaly-based alerting** — a fixed static bound ("P95 latency > 30s") versus a comparison against a learned baseline ("P95 latency is statistically different from its trailing average"). You've already built one of each without necessarily naming them that way — today you name them and choose deliberately, signal by signal.
2. **Alert fatigue** — why alerting on every single data point is actively harmful, not just noisy: real incidents get lost in the noise once people learn to ignore or mute a chronically over-triggering channel. Fewer, higher-precision alerts beat many low-precision ones.
3. **What deserves a 3am page** — a severity framework distinguishing "wake someone up, this is broken and blocking users" from "make a ticket, look at it tomorrow" from "log it, no human needs to see this unless they go looking." Getting this wrong in either direction is costly — under-alerting misses real incidents, over-alerting trains people to ignore the pager entirely.

---

## Alert Fatigue, Made Concrete With Your Own Data

Recall 2.3's finding: LLM latency is heavy-tailed. A meaningful fraction of requests will always be slower than the median, just because of how generation length varies. If you alerted on _every single request_ that exceeded, say, the P90 latency of its own distribution — you would page constantly, by definition, because roughly 10% of all requests always sit above the P90 by construction. That's not a monitoring system catching real problems; it's noise dressed up as signal.

```
BAD: Alert on every individual slow request
  Request 1: 4.2s   → fine
  Request 2: 22.1s  → ALERT (but this might just be a long answer)
  Request 3: 3.8s   → fine
  Request 4: 19.4s  → ALERT (again — normal variance, not an incident)
  ...
  → Constant noise. Within a week, someone mutes the channel.
  → The one time latency ACTUALLY indicates a real problem, nobody notices.

GOOD: Alert on a sustained aggregate crossing a threshold
  P95 latency over a rolling 1-hour window > 30s, for the WHOLE window
  → Fires rarely. When it fires, it means something.
  → This is what 2.3's percentile machinery was for all along.
```

This is the entire argument for everything you already built in 2.2, 2.3, and 2.5: percentiles over windows, not single data points; statistical comparisons against a baseline, not raw eyeballing. Today's job is applying that same discipline to _when something should interrupt a human_, not just to what you measure.

---

## Threshold vs Anomaly-Based — You've Already Built Both

|                           | Threshold-based                                                                  | Anomaly-based                                                                      |
| ------------------------- | -------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| Definition                | A fixed, static numeric bound                                                    | Statistical comparison against a learned/trailing baseline                         |
| Example you already built | 2.2's preview Monitor: latency > 20000ms                                         | 2.5's drift detector: current window vs baseline window, z-test                    |
| Strength                  | Simple, predictable, easy to reason about                                        | Adapts to your system's own normal patterns                                        |
| Weakness                  | Doesn't adapt — a threshold right for Tuesday might be wrong for a traffic spike | More complex; needs enough historical data for the baseline to mean anything       |
| Best for                  | Signals with a real, known "broken" line (error rate, hard cost budget)          | Signals where "normal" varies and what matters is _change_, not an absolute number |

You didn't need new theory today for this — you needed the vocabulary for what you'd already built. 2.2's Monitor was threshold. 2.5's drift detector was anomaly-based. Today decides, for each of four signals, which one fits — and in error rate's case specifically, why the answer is "threshold, deliberately, even though a statistical comparison is available too."

---

## Part A: Add Error Rate — the One Signal You Don't Have Yet

Nothing in 2.1–2.6 explicitly tracked failures. Fix that first, since it's the most important "3am page" candidate of all four signals — an elevated error rate means the system is _actually broken_, not just slow or drifting.

### Extend the Schema

```python
# app/db.py — add to init_request_log_table()
async def init_request_log_table():
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS request_log (
                id SERIAL PRIMARY KEY,
                trace_id TEXT,
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
        # Migrations for tables that already existed from Session 2.5 —
        # idempotent, safe to run every startup.
        await conn.execute("ALTER TABLE request_log ALTER COLUMN trace_id DROP NOT NULL")
        await conn.execute("ALTER TABLE request_log ADD COLUMN IF NOT EXISTS is_error BOOLEAN NOT NULL DEFAULT FALSE")
        await conn.execute("ALTER TABLE request_log ADD COLUMN IF NOT EXISTS error_message TEXT")
```

`trace_id` drops its `NOT NULL` constraint because a failed request may never reach the point where a trace ID is available to log.

### Catch, Log, and Surface Failures in `/query`

```python
from fastapi import HTTPException

@app.get("/query")
async def query_endpoint(
    q: str = QueryParam(..., min_length=3),
    user_id: str = QueryParam(default="anonymous"),
    prompt_version: int | None = QueryParam(default=None),
):
    pool = await get_pool()
    try:
        result = run_rag_pipeline(query=q, user_id=user_id, prompt_version=prompt_version)
        scores = score_answer(result["trace_id"], result["answer"])

        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO request_log
                    (trace_id, user_id, query, query_length, answer_length,
                     cites_source, reasonable_length, prompt_version, is_error)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, FALSE)
                """,
                result["trace_id"], user_id, q, len(q), len(result["answer"]),
                scores["cites_source"], scores["reasonable_length"],
                str(prompt_version) if prompt_version else "production",
            )
        return result

    except Exception as exc:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO request_log (user_id, query, query_length, is_error, error_message)
                VALUES ($1, $2, $3, TRUE, $4)
                """,
                user_id, q, len(q), str(exc)[:500],
            )
        raise HTTPException(status_code=502, detail="Upstream generation failed")
```

### Add Error Rate to `/admin/drift-report`

Extend the existing endpoint rather than adding a parallel one — a real engineer evolves an existing report incrementally instead of fragmenting monitoring across near-duplicate endpoints:

```python
@app.get("/admin/drift-report")
async def drift_report(
    baseline_n: int = QueryParam(default=50, ge=5),
    current_n: int = QueryParam(default=50, ge=5),
):
    pool = await get_pool()
    async with pool.acquire() as conn:
        baseline_rows = await conn.fetch(
            "SELECT query_length, cites_source, is_error FROM request_log ORDER BY created_at ASC LIMIT $1",
            baseline_n,
        )
        current_rows = await conn.fetch(
            "SELECT query_length, cites_source, is_error FROM request_log ORDER BY created_at DESC LIMIT $1",
            current_n,
        )

    if len(baseline_rows) < 5 or len(current_rows) < 5:
        return {"error": "Not enough data yet — need at least 5 rows in each window."}

    # Error rate uses ALL rows, including failures.
    baseline_errors = sum(1 for r in baseline_rows if r["is_error"])
    current_errors = sum(1 for r in current_rows if r["is_error"])
    error_z, error_p = two_proportion_z_test(baseline_errors, len(baseline_rows), current_errors, len(current_rows))
    current_error_rate = current_errors / len(current_rows)

    # Input drift and output quality drift are only meaningful on
    # successful requests — a crashed request never got a chance to cite
    # anything, and including it would conflate "broken" with "ungrounded."
    baseline_ok = [r for r in baseline_rows if not r["is_error"]]
    current_ok = [r for r in current_rows if not r["is_error"]]

    baseline_lengths = [r["query_length"] for r in baseline_ok]
    current_lengths = [r["query_length"] for r in current_ok]
    length_z, length_p = two_sample_z_test_means(baseline_lengths, current_lengths)

    baseline_cites = sum(1 for r in baseline_ok if r["cites_source"])
    current_cites = sum(1 for r in current_ok if r["cites_source"])
    cite_z, cite_p = two_proportion_z_test(baseline_cites, len(baseline_ok), current_cites, len(current_ok))

    return {
        "reliability": {
            "baseline_error_rate": round(baseline_errors / len(baseline_rows), 3),
            "current_error_rate": round(current_error_rate, 3),
            "z": round(error_z, 2),
            "p": round(error_p, 4),
            # Deliberately an ABSOLUTE floor, not just the z-test — see
            # "Why Error Rate Is Threshold-Based" below.
            "flagged": current_error_rate > 0.05,
        },
        "input_drift": {
            "baseline_mean_length": round(statistics.mean(baseline_lengths), 1) if baseline_lengths else None,
            "current_mean_length": round(statistics.mean(current_lengths), 1) if current_lengths else None,
            "z": round(length_z, 2),
            "p": round(length_p, 4),
            "flagged": length_p < 0.05,
        },
        "output_quality_drift": {
            "baseline_cite_rate": round(baseline_cites / len(baseline_ok), 3) if baseline_ok else None,
            "current_cite_rate": round(current_cites / len(current_ok), 3) if current_ok else None,
            "z": round(cite_z, 2),
            "p": round(cite_p, 4),
            "flagged": cite_p < 0.05,
        },
    }
```

### Why Error Rate Is Threshold-Based, Deliberately

A z-test asks "is this different from baseline?" But if your baseline _already_ had a 4% error rate — maybe it always has, maybe nobody noticed — a z-test comparing 4% to 4.5% might not reach significance, and you'd never page on it. **An elevated absolute error rate is bad regardless of what "normal" has been for you.** This is exactly the case threshold-based alerting exists for: some things have a real, known "broken" line independent of your own history. `current_error_rate > 0.05` gates the page here; the z-test comparison is informational, not the trigger.

---

## Part B: Threshold Alerts via Langfuse Monitors (Cloud-Side)

**Why these work regardless of whether your laptop is on:** Langfuse Monitors evaluate against data already ingested into Langfuse Cloud — they don't need a live connection back to your machine. This matters, and it's the key distinction the rest of this session hinges on (see Part C).

Set up a free notification channel first — **Slack incoming webhooks** are Langfuse's most directly documented option: create an app at [api.slack.com/apps](https://api.slack.com/apps) → Incoming Webhooks → Activate → Add New Webhook to Workspace → copy the URL. A generic webhook target is also supported if you'd rather not set up Slack — check the exact payload format in the Monitors UI when you configure it.

Build three monitors in **Langfuse → Monitors → New Monitor**:

| Monitor              | Data Source  | Metric        | Threshold    | Window   |
| -------------------- | ------------ | ------------- | ------------ | -------- |
| P95 latency          | Observations | p95 latency   | > 30000 (ms) | 1 hour   |
| Daily cost           | Observations | sum totalCost | > $0.01      | 24 hours |
| Request volume floor | Observations | count         | < 1          | 1 hour   |

**On the cost threshold being tiny:** with local phi3:mini inference, real cost is $0 unless you added 2.2's reference pricing — this monitor is currently symbolic, proving the mechanism works rather than protecting a real budget. The moment Pillar 4 or 12 points this pipeline at a paid API, this exact monitor becomes load-bearing; you'd just raise the threshold to match a real budget number. Build the muscle now, while the stakes are zero.

**On the volume floor being a _minimum_, not a maximum:** most of your instincts so far have been "alert when something goes too high." A silent outage — your API becoming unreachable, Ollama crashing, the process dying — doesn't show up as elevated _anything_; it shows up as **volume dropping to zero**. Error-rate monitoring only catches requests that were attempted and failed; it says nothing about requests that never arrived because nothing was there to receive them. A volume floor is a distinct, necessary signal — not redundant with error rate.

---

## Part C: Anomaly Alerts — Why These Can't Be a Cloud-Scheduled Job

Here's the constraint worth understanding precisely, not working around quietly: `/admin/drift-report` reads from **your local Postgres `request_log` table**. A GitHub Actions scheduled workflow runs on a GitHub-hosted VM in GitHub's cloud — it has no path to `http://localhost:8000` on your personal laptop, full stop. This is fundamentally different from 2.6's CI eval gate, which works precisely _because_ it spins up its own complete, self-contained stack inside the CI job (its own Ollama, calling the pipeline function directly) rather than depending on reaching your machine.

**The honest options for local-only infrastructure:**

```
┌─────────────────────────────────────────────────────────────────┐
│ Can automate in the cloud:                                       │
│   Anything reading data already SENT to a cloud service          │
│   (Langfuse Monitors — Part B)                                   │
│   Anything that spins up its own complete stack inside the job   │
│   (2.6's CI eval gate)                                            │
├─────────────────────────────────────────────────────────────────┤
│ Can only automate locally, and only while the machine is on:     │
│   Anything reading data that ONLY exists on your machine          │
│   (drift-report's Postgres query — this section)                 │
│   → Local script, run manually or via OS-level scheduling         │
│   → NOT a real substitute for always-on monitoring — a deployed   │
│     service is what that actually requires, and that's out of    │
│     scope without a cloud budget                                  │
└─────────────────────────────────────────────────────────────────┘
```

Naming this limitation clearly is itself a legitimate engineering skill — knowing which of your monitors are genuinely always-on versus which quietly depend on your laptop being powered on and your app running is exactly the kind of gap that bites teams in production. Better to know it now.

### The Local Check Script

```python
# scripts/check_drift_alert.py
"""
Checks the LOCAL /admin/drift-report and posts a Slack alert if anything
is flagged. Talks to localhost because it runs on the same machine as the
app — this is NOT a substitute for cloud-based always-on monitoring, see
Part C notes on why that would require a deployed service.
"""
import os
import requests
from dotenv import load_dotenv
load_dotenv()

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
API_BASE = "http://localhost:8000"


def check_and_alert():
    resp = requests.get(f"{API_BASE}/admin/drift-report", params={"baseline_n": 50, "current_n": 20})
    resp.raise_for_status()
    report = resp.json()

    if "error" in report:
        print(report["error"])
        return

    flags = []
    if report["reliability"]["flagged"]:
        flags.append(
            f"🔴 Error rate: {report['reliability']['current_error_rate']:.1%} "
            f"(baseline {report['reliability']['baseline_error_rate']:.1%})"
        )
    if report["output_quality_drift"]["flagged"]:
        flags.append(
            f"🟡 Output quality drift: citation rate "
            f"{report['output_quality_drift']['current_cite_rate']:.1%} vs "
            f"baseline {report['output_quality_drift']['baseline_cite_rate']:.1%}"
        )
    if report["input_drift"]["flagged"]:
        flags.append(
            f"ℹ️ Input drift: mean query length "
            f"{report['input_drift']['current_mean_length']} vs "
            f"baseline {report['input_drift']['baseline_mean_length']}"
        )

    if not flags:
        print("✓ No drift flagged")
        return

    message = "*Drift Alert — ai-infra-portfolio*\n" + "\n".join(flags)
    print(message)

    if SLACK_WEBHOOK_URL:
        requests.post(SLACK_WEBHOOK_URL, json={"text": message})
    else:
        print("(SLACK_WEBHOOK_URL not set — printed only, not sent)")


if __name__ == "__main__":
    check_and_alert()
```

```bash
pip install requests   # add to requirements.txt if not already present
python scripts/check_drift_alert.py
```

### Optional: Windows Task Scheduler

To run this on a recurring basis **while your laptop is on and the app is running** (not a substitute for a deployed service, but a reasonable local approximation):

```powershell
schtasks /create /tn "DriftCheck" /tr "python C:\path\to\ai-infra-portfolio\scripts\check_drift_alert.py" /sc minute /mo 30
```

This fires every 30 minutes, only while Windows is running the task scheduler (i.e., machine on) and only meaningfully while your FastAPI app and Ollama are also running. Worth setting up once to see it work — not worth treating as production-grade monitoring.

---

## Part D: The Alert Matrix — the Actual Design Deliverable

Create `docs/alerting.md` in your repo:

```
### Alert Matrix — ai-infra-portfolio

| Signal | Category | Type | Where It Runs | Threshold | Severity | Disposition |
|---|---|---|---|---|---|---|
| CI eval gate | Pre-merge | Absolute floor | GitHub Actions | pass rate < 83% | Blocks merge | Never a page — blocks shipping instead |
| Error rate | Runtime | Threshold (absolute) | Local script | > 5% of last 20 requests | Critical | 3am page — system is actually broken |
| Request volume | Runtime | Threshold (floor) | Langfuse Monitor | count < 1 / hour | Critical | 3am page — possible total outage |
| P95 latency | Runtime | Threshold | Langfuse Monitor | > 30s, sustained 1hr | Warning | Next business day — slow, not down |
| Daily cost | Runtime | Threshold | Langfuse Monitor | > budget (symbolic today) | Warning | Next business day — becomes real post-Pillar 4/12 |
| Output quality drift | Runtime | Anomaly (z-test) | Local script | p < 0.05, sustained 2 checks | Warning → Critical if sustained | Same-day investigate; escalate if persistent |
| Input (data) drift | Runtime | Anomaly (z-test) | Local script | p < 0.05 | Informational | Dashboard/log only — informs golden set updates, not an incident |

```

## Why these dispositions

- **Error rate and volume floor page immediately** — both indicate the
  system is unusable _right now_, with no workaround available to users.
- **Latency and cost wait for business hours** — annoying, budget-relevant,
  never actually blocking someone from getting _an_ answer.
- **Output quality drift escalates only if sustained** — a single flagged
  window could be noise (Session 2.5/2.6's small-sample lesson). Requiring
  two consecutive flagged checks before treating it as urgent is a direct
  alert-fatigue mitigation: it trades a few hours of detection latency for
  a large reduction in false pages.
- **Input drift never pages at all** — a shift in what people are asking
  isn't inherently bad; it's information that your golden set (2.6) or
  prompt (2.4) might need revisiting. It belongs in a dashboard someone
  checks periodically, not in anyone's pocket at 3am.

This file is a real, defensible portfolio artifact — it demonstrates you can reason about severity, not just wire up a metric.

---

## Common Failure Modes

### Failure 1: Langfuse Monitor never fires, even during a genuine slow period

**Diagnose:** confirm the monitor's data source and filter actually match what you're generating — a monitor scoped to `observations` with no name filter aggregates across retrieval, context*assembly, \_and* generation together, which can dilute a real generation-latency spike. Consider filtering to the specific observation `name` if the aggregate isn't sensitive enough.

### Failure 2: `check_drift_alert.py` gets a connection error

**Root Cause:** almost always that the FastAPI app isn't currently running — this script depends on `localhost:8000` being live, unlike the CI eval gate, which is fully self-contained. **Fix:** confirm `uvicorn` is running before invoking the script, or before a scheduled Task Scheduler run fires.

### Failure 3: Everything is flagged as "critical," all the time

**Root Cause:** thresholds set from a single anecdotal bad experience rather than your own measured baseline. **Fix:** before finalizing any threshold in `docs/alerting.md`, pull your own actual P50/P95 numbers from 2.3's `/admin/latency-report` and set thresholds meaningfully above your observed normal — not a round number that felt right. A latency threshold set below your normal P95 will page constantly and teach you (or a future teammate) to ignore it, which is the alert-fatigue failure mode this entire session exists to prevent.

---

## Verification Checklist

- [x] `request_log` migrated with `is_error` and `error_message`, `trace_id` nullable
- [x] `/query` catches exceptions, logs them, returns a proper `502` instead of an unhandled crash
- [x] `/admin/drift-report` now includes a `reliability` section with an absolute-threshold flag
- [x] Three Langfuse Monitors created: P95 latency, daily cost, request volume floor
- [x] `scripts/check_drift_alert.py` runs locally and correctly reports "no drift flagged" on healthy data
- [x] Deliberately triggered an error (e.g., stop Ollama, send a request) and confirmed it's logged and reflected in `reliability.current_error_rate`
- [x] `docs/alerting.md` committed, covering all four runtime signals plus the CI gate
- [x] You can explain, without notes, why error rate uses an absolute threshold instead of a z-test comparison
- [x] You can explain why the drift-check script can't be a GitHub Actions scheduled job, precisely

---

## Directory Structure After This Session

```
ai-infra-portfolio/
├── app/
│   ├── main.py               ← /query error handling; /admin/drift-report extended with reliability
│   ├── db.py                  ← request_log migrated: is_error, error_message, nullable trace_id
│   ├── instrumentation.py
│   ├── llm_client.py
│   └── rag_pipeline.py
├── eval/
├── scripts/
│   ├── setup_prompts.py
│   ├── run_prompt_ab_test.py
│   └── check_drift_alert.py   ← NEW
├── docs/
│   └── alerting.md             ← NEW — the alert matrix design artifact
├── .github/workflows/eval.yml
├── .env                        ← SLACK_WEBHOOK_URL added
└── requirements.txt             ← requests added
```

---

## Evening Integration — Before You Close the Laptop

```bash
git add app/main.py app/db.py scripts/check_drift_alert.py docs/alerting.md requirements.txt
git commit -m "[pillar-2.7] error rate tracking, Langfuse Monitors, local drift alerting, alert matrix design"
git push
```

**3-sentence journal:**

1. What you built today and what it does
2. What broke and the exact error / symptom
3. What you'd do differently if starting over

---

## Commit Message

```
[pillar-2.7] error rate tracking, Langfuse Monitors, local drift alerting, alert matrix design
```

---

## Pillar 2 Is Complete

Seven sessions ago, `/query` returned an answer with zero visibility into what happened inside it. Now: every request is traced end-to-end (2.1), cost-attributed per user (2.2), latency-profiled down to TTFT/TPOT (2.3), running a versioned, experimentally-validated prompt (2.4), watched for drift in both its inputs and its outputs (2.5), gated by an automated regression test before anything ships (2.6), and wired into a real, severity-tiered alert matrix (2.7). This is Pillar 2's whole promise — **the primary differentiator** for a junior candidate — and it's done.

One thread has been pulled through every single session as a deliberate, named limitation: retrieval has been three hardcoded mock chunks since 2.1. Every "this will be real in Pillar 10" note across these seven sessions was pointing at the same thing. **That's next.** Pillar 10 rips out the mock and puts in Qdrant, real chunking, real hybrid search — and everything you built in Pillar 2 keeps working underneath it, unchanged, because you built your observability layer around the _shape_ of the pipeline, not around the specific fake data flowing through it today.

---

## Quick Reference Card

```
ALERTING TAXONOMY:
  Threshold-based  = fixed static bound (error rate, cost budget, volume floor)
  Anomaly-based    = comparison against a learned baseline (drift, sustained latency shift)
  Pre-merge gate   = neither — blocks shipping, never pages anyone (2.6)

WHAT PAGES AT 3AM:
  Error rate spike, request volume collapse — system unusable, no workaround

WHAT WAITS FOR BUSINESS HOURS:
  Elevated latency, cost creep - degraded, not broken

WHAT'S INFORMATIONAL ONLY:
  Input/data drift alone — not inherently bad, just worth reviewing periodically

ALERT FATIGUE MITIGATIONS USED TODAY:
  Aggregate over a window, never alert on one data point (carried from 2.3)
  Require SUSTAINED signal (2+ consecutive checks) before escalating severity
  Absolute floor for error rate — don't let a z-test hide an already-bad baseline

CLOUD VS LOCAL AUTOMATION — THE CORE DISTINCTION:
  Langfuse Monitors      → work regardless of your laptop, data already in the cloud
  CI eval gate (2.6)     → works because CI spins up its OWN full stack
  Local drift script     → only works while YOUR machine + app are running —
                            not a substitute for a deployed, always-on service
```
