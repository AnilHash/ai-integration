
|||
|---|---|
|**Pillar**|2 — Observability (Tier 1, ⭐ primary differentiator)|
|**Prereqs**|2.1 (Langfuse tracing wired in), 2.2 (token counting on every call)|
|**Feeds into**|2.4 (prompt A/B testing needs a latency baseline to detect regressions)|
|**Where**|Local — CPU inference on the Ryzen 5 3500U, no GPU needed|
|**SDKs verified for this doc**|`langfuse` Python SDK v4 (rewritten March 2026 — if `pip show langfuse` gives you a 2.x, upgrade before starting), `ollama` Python library (current as of July 2026)|

---

## 1. Session Goal

You're going to stop treating "the LLM call" as one black box and split it into the pieces that actually cost time: retrieval, prompt assembly, time-to-first-token, and token-by-token generation. By the end you'll have real P50/P95/P99 numbers from your own portfolio pipeline and know exactly which stage is your bottleneck — not a guess, a measurement.

---

## 2. The 3 Concepts You Must Be Able to Explain Without Notes

1. **TTFT vs TPOT vs E2E** — what each measures, why they're driven by different hardware bottlenecks, and how they combine into total latency.
2. **P50/P95/P99 thinking** — why a single average latency number actively hides the problems you're being hired to catch.
3. **Where time actually goes in a RAG request** — the stage-by-stage breakdown methodology, and how to instrument it so the answer is a number, not a guess.

---

## 3. TTFT vs TPOT vs E2E

```
 0ms                                                                        E2E
 |───────────────┬────────────────┬─────────────────┬───────────────────────|
 │   Retrieval    │  Prompt build  │  TTFT (prefill)  │  Generation (TPOT×n)  │
 │  (vector/stub) │ (context+tmpl) │  time to 1st tok │  streamed to client   │
 └────────────────┴────────────────┴──────────────────┴───────────────────────┘
                                     ▲
                                     │
                         user perceives latency ends here
                         if you're streaming — this is why
                         streaming UIs *feel* faster even when
                         E2E is identical
```

**TTFT (Time To First Token)** — wall-clock time from "request received" to "first token of the response exists." Dominated by: queue wait, retrieval, prompt assembly, and the model's **prefill** pass (processing the entire input prompt through the network once, in parallel, before any generation starts). Prefill cost scales with **prompt length**, not output length.

**TPOT (Time Per Output Token)** — once generation starts, how long each subsequent token takes. This is the **decode** phase: one token in, one token out, repeated. On CPU inference (your Vega 8 has no discrete VRAM, so `phi3:mini` runs on the CPU) this is memory-bandwidth bound, not compute bound — you're moving the model's weights through the CPU cache/RAM path for every single token.

**E2E (End-to-End) latency** — everything, start to finish:

```
E2E ≈ network_in + queue_wait + retrieval + prompt_assembly
      + TTFT
      + TPOT × (tokens_out − 1)
      + response_formatting + network_out
```

This is why "average latency: 4.2s" is a useless headline metric on its own — it doesn't tell you whether you have a slow retrieval step, a slow prefill, or a model that's just generating 400 tokens when 80 would do. Splitting E2E into these components is the entire point of this session.

**Worked example** (numbers you should replace with your own measurements — don't trust mine): `phi3:mini` on your Ryzen 5 3500U runs roughly 15–30 tok/s once warm, so TPOT ≈ 33–66ms/token. A 150-token response is then `150 × ~45ms ≈ 6.75s` of pure generation, on top of whatever TTFT you measure for prefill. If your E2E comes back at 7.0s and TTFT is 1.2s, generation is clearly your dominant cost — not retrieval, not the network. That's the kind of statement you should be able to make about your own pipeline after this lab, backed by a number.

---

## 4. P50/P95/P99 Thinking

```
Latency distribution — 50 requests, phi3:mini, CPU

 count
   │                         ▄▄
   │                       ▄████▄
   │                     ▄████████▄
   │                   ▄██████████████▄
   │                 ▄██████████████████▄▄
   │_______________▄██████████████████████▄▄___________▄▄___
   └───────────────┴──────┴───────┴────────┴─────────────┴──► latency
                   P50    P75     P90      P95            P99
                  "typical"                "rare"      "OS scheduling hiccup,
                                                          thermal throttle,
                                                          background process
                                                          stole your CPU"
```

Your MERN background already has the instinct for this — you'd never ship an Express API to production and only look at average response time; you'd load-test with `autocannon` or `k6` and check the tail. Same discipline applies here, and it matters _more_ for LLM inference because CPU-bound token generation on a shared laptop is far noisier than a typical REST endpoint: background OS processes, thermal throttling on an integrated GPU that shares system RAM, and model-load cold starts all show up as tail latency, not average latency.

**Why average lies:** if 45 of 50 requests take 4s and 5 take 20s (because Ollama had to reload the model, or Windows decided to run a background update), your average is ~5.6s — looking totally fine — while 10% of your real users are sitting through a 20-second wait. P95/P99 is what catches that. This is also literally an interview question (see §8).

---

## 5. Where Time Actually Goes — The Stage Model

Map this onto Langfuse's trace/span/generation model directly:

```
TRACE: POST /query  (root)
 │
 ├─ SPAN (as_type="retriever"): retrieval_stage
 │    └─ stub for now — real Qdrant integration lands in Pillar 10
 │
 ├─ SPAN (as_type="span"): prompt_assembly
 │
 ├─ GENERATION (as_type="generation"): llm_call  — Ollama · phi3:mini
 │    ├─ metadata.ttft_ms                 (wall-clock, measured by you)
 │    ├─ metadata.tpot_ms_ollama_reported (derived from Ollama's own timers)
 │    ├─ metadata.prompt_eval_duration_ms (Ollama-reported prefill time)
 │    └─ metadata.eval_duration_ms        (Ollama-reported generation time)
 │
 └─ SPAN (as_type="span"): response_formatting
```

**Honest note on retrieval:** Qdrant doesn't exist in your portfolio project yet — it arrives in Pillar 10. Whatever your "retrieval" stage is right now (in-memory cosine similarity over a handful of embedded chunks, or a stub function that just returns a placeholder) is fine for this lab. The point of 2.3 is the _profiling methodology and instrumentation_, not retrieval quality. You'll re-profile retrieval for real once Qdrant is wired in and you're comparing HNSW vs IVF in 10.1.

---

## 6. MERN Bridge

|MERN/Node concept you already know|AI infra equivalent in this session|
|---|---|
|First-byte time on an Express SSE/streaming response|TTFT|
|Throughput of a stream after headers are already sent|TPOT|
|p95/p99 dashboards you'd build with `autocannon` or `k6` against an Express API|Same percentile discipline, applied to LLM calls|
|Blocking the Node event loop with `fs.readFileSync()` inside a request handler|Blocking FastAPI's event loop with a **sync** Ollama client inside `async def` (see Failure 2 below)|
|`app.use()` middleware chain, each middleware doing its bit before calling `next()`|Nested Langfuse spans inside one trace, each stage closing before the next opens|

---

## 7. Lab Setup

### 7.1 Environment check (do this first — do not skip)

`load_dotenv()` must execute **before** anything constructs the Langfuse client, because the client reads credentials at first call, not lazily per-request. FastAPI/uvicorn does **not** auto-load `.env` files — that's a Node/`dotenv` habit that doesn't carry over silently.

```python
# top of app/main.py — import order matters
from dotenv import load_dotenv
load_dotenv()          # MUST run before `from langfuse import get_client` is *called*

import os
print("LANGFUSE_PUBLIC_KEY set:", bool(os.getenv("LANGFUSE_PUBLIC_KEY")))
print("LANGFUSE_BASE_URL:", os.getenv("LANGFUSE_BASE_URL"))
```

Run that diagnostic snippet standalone first. If either line prints `False` / `None`, fix your `.env` before writing a single line of instrumentation — a silently-disconnected Langfuse client will make this whole lab look broken when it isn't.

Your `.env` (Langfuse Cloud free tier, per project rules — never self-hosted v3 on this hardware):

```
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

### 7.2 Instrument every stage

`app/main.py`:

```python
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel
from langfuse import get_client
from ollama import AsyncClient

langfuse = get_client()
ollama_client = AsyncClient()          # ASYNC client — see Failure 2 for why this matters


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    langfuse.flush()                   # don't lose buffered spans on shutdown


app = FastAPI(lifespan=lifespan)


class QueryRequest(BaseModel):
    query: str


def retrieve_stub(query: str) -> list[str]:
    # Placeholder until Pillar 10 (Qdrant). This lab profiles the SHAPE
    # of the pipeline, not retrieval quality.
    return [f"[stub context chunk for: {query[:30]}]"]


@app.post("/query")
async def query(req: QueryRequest):
    with langfuse.start_as_current_observation(as_type="span", name="query-pipeline") as trace_span:

        t0 = time.perf_counter()
        with langfuse.start_as_current_observation(as_type="retriever", name="retrieval_stage") as r_span:
            context = retrieve_stub(req.query)
            r_span.update(output={"chunks": len(context)})
        t1 = time.perf_counter()

        with langfuse.start_as_current_observation(as_type="span", name="prompt_assembly") as p_span:
            prompt = f"Context:\n{context}\n\nQuestion: {req.query}"
            p_span.update(output={"prompt_chars": len(prompt)})
        t2 = time.perf_counter()

        ttft = None
        tokens_out = 0
        full_response = ""
        final_meta = {}

        gen_start = time.perf_counter()
        with langfuse.start_as_current_observation(
            as_type="generation", name="llm_call", model="phi3:mini"
        ) as gen_span:

            stream = await ollama_client.chat(
                model="phi3:mini",
                messages=[{"role": "user", "content": prompt}],
                stream=True,             # <-- if this is missing, TTFT == E2E. See Failure 1.
            )

            async for chunk in stream:
                content = chunk["message"]["content"]
                if ttft is None and content:
                    ttft = time.perf_counter() - gen_start
                if content:
                    full_response += content
                    tokens_out += 1
                if chunk["done"]:
                    final_meta = {
                        "prompt_eval_duration_ms": chunk["prompt_eval_duration"] / 1e6,
                        "eval_duration_ms": chunk["eval_duration"] / 1e6,
                        "eval_count": chunk["eval_count"],
                    }

            gen_end = time.perf_counter()
            tpot_reported = (
                final_meta["eval_duration_ms"] / final_meta["eval_count"]
                if final_meta.get("eval_count") else None
            )

            gen_span.update(
                output=full_response,
                metadata={
                    "ttft_ms": round(ttft * 1000, 1) if ttft else None,
                    "tpot_ms_ollama_reported": round(tpot_reported, 1) if tpot_reported else None,
                    **final_meta,
                },
            )
        t3 = time.perf_counter()

        trace_span.update(
            metadata={
                "retrieval_ms": round((t1 - t0) * 1000, 1),
                "prompt_assembly_ms": round((t2 - t1) * 1000, 1),
                "llm_call_ms": round((t3 - gen_start) * 1000, 1),
                "e2e_ms": round((t3 - t0) * 1000, 1),
            }
        )

    return {
        "response": full_response,
        "e2e_ms": round((t3 - t0) * 1000, 1),
        "ttft_ms": round(ttft * 1000, 1) if ttft else None,
    }
```

**Two timing sources, on purpose:**

- Your own `time.perf_counter()` TTFT is the _real user-experienced_ number — it includes retrieval, prompt assembly, Ollama's HTTP round trip, everything.
- Ollama's self-reported `prompt_eval_duration` / `eval_duration` / `eval_count` (returned in nanoseconds on the final streamed chunk) tell you specifically how much of that time was the model's own prefill vs decode — useful for deciding whether a slow request is a prompt-length problem or a generation-length problem.

### 7.3 Run 30–50 requests and compute percentiles

Five requests is not a sample size — see Failure 3. `scripts/latency_load_test.py`:

```python
import asyncio
import time

import httpx
import numpy as np

N_REQUESTS = 40
URL = "http://localhost:8000/query"
QUERIES = [f"What is idempotency? (run {i})" for i in range(N_REQUESTS)]


async def run():
    e2e_ms, ttft_ms = [], []
    async with httpx.AsyncClient(timeout=60) as client:
        for q in QUERIES:                      # sequential first — concurrency is a separate test
            t0 = time.perf_counter()
            resp = await client.post(URL, json={"query": q})
            t1 = time.perf_counter()
            data = resp.json()
            e2e_ms.append((t1 - t0) * 1000)
            if data.get("ttft_ms"):
                ttft_ms.append(data["ttft_ms"])

    def report(name, values):
        arr = np.array(values)
        print(f"\n{name} (n={len(arr)})")
        print(f"  P50: {np.percentile(arr, 50):.0f}ms")
        print(f"  P95: {np.percentile(arr, 95):.0f}ms")
        print(f"  P99: {np.percentile(arr, 99):.0f}ms")
        print(f"  mean: {arr.mean():.0f}ms   max: {arr.max():.0f}ms")

    report("E2E latency", e2e_ms)
    report("TTFT", ttft_ms)


if __name__ == "__main__":
    asyncio.run(run())
```

`pip install numpy httpx --break-system-packages` if you don't already have them.

**Verify:** you should walk away with four real numbers — P50/P95/P99 for both E2E and TTFT — plus, from your Langfuse dashboard, a breakdown showing what fraction of E2E is retrieval vs prompt assembly vs TTFT vs pure generation for a handful of individual traces. That breakdown is the actual deliverable of this session — the load test script just gives you volume.

### 7.4 Optional but instructive — concurrency test

Modify the load test to fire 5 requests concurrently with `asyncio.gather` instead of the sequential loop. If your P95 scales roughly linearly with concurrency instead of staying flat, that's your evidence for Failure 2 below — walk through the diagnosis there before you conclude it's "just CPU inference being slow."

### 7.5 RAM/Docker check

No new containers this session — you're still on FastAPI + Postgres (compose) + Ollama (native) from Pre-Work. Run `docker stats --no-stream` before the lab anyway; if you've left anything running from a previous session, close it. This session doesn't push you near the 5.92GB ceiling on its own.

---

## 8. Common Failures

**Failure 1 — Forgot `stream=True`, so TTFT literally equals E2E** Call `ollama_client.chat(...)` without `stream=True` and the call blocks until the _entire_ response is generated before returning anything — there's no "first token" event to time. If you compute TTFT as "time until the response object exists," you'll get a number identical to E2E, and every conclusion downstream is wrong. **Diagnose:** grep your code for `stream=True`. If it's missing, that's the bug — not your model, not your hardware.

**Failure 2 — Sync Ollama client inside `async def` blocks the event loop** Using `ollama.Client()` (sync) instead of `ollama.AsyncClient()` inside a FastAPI async route blocks the entire event loop for the full duration of generation — nothing else can be served, exactly like calling `fs.readFileSync()` synchronously inside an Express handler. **Diagnose:** fire 3 concurrent requests with `asyncio.gather` (§7.4). If P95 scales up roughly linearly with concurrency instead of staying close to flat, you have a blocking call somewhere in the request path — check that every I/O call (Ollama, Postgres, anything) uses its async client.

**Failure 3 — P95/P99 computed from too few samples** At N=5, your "P95" is just your 5th-highest value relabeled — it's noise, not a percentile. **Diagnose:** run the load test at N=10, then N=50. If P95 moves by more than ~20% between the two, you didn't have enough samples the first time. Treat anything under N=30 as unreliable for tail-latency claims.

---

## 9. Interview Readiness Tie-In

This session directly answers: _"What's TTFT and why does it matter for user experience?"_ from your Tier 1 checklist.

**Junior answer:** "TTFT is how long until the first token comes back."

**Senior answer:** "TTFT is prefill-bound — it scales with prompt length, not output length — while the rest of generation is decode-bound and scales with output length and TPOT. Streaming UIs hide TPOT from the user's perceived wait but not from the server's resource cost, so I profile both separately: TTFT tells me if retrieval or prompt size is the problem, TPOT tells me if it's raw generation throughput. I built the split with Langfuse spans on my portfolio project — here's the GitHub." Close every answer that way.

---

## 10. Commit Message

```
[pillar-2.3] add latency profiling spans + TTFT/TPOT/percentile analysis to RAG pipeline
```

---

## 11. What's Next

**2.4 — Prompt Versioning & A/B Testing.** You'll need today's latency baseline: when you version two prompts and compare them, you're checking quality _and_ making sure you didn't silently double your P95 by making the prompt twice as long.