## Retrieval: Chunk Size Evaluation

Tested recall@5/MRR/NDCG@5 across chunk sizes 256/512/1024 using nomic-embed-text-v1.5. Default chunk size: 256 (ties on recall, wins on MRR vs 512; 1024 degrades sharply).

Found and fixed two bugs during evaluation:

- Initial run used all-MiniLM-L6-v2, which silently truncates at 256 tokens — made 1024-chunk recall look like a chunking problem when it was actually an embedding-truncation artifact.
- Missed nomic's asymmetric encoding prefixes (search_document / search_query) on first pass with the new model, which suppressed recall further.
- Also hit a dimension mismatch after swapping embedding models — Qdrant collection was still configured for 384-dim (MiniLM) after switching to a 768-dim model. Had to delete the volume/collection and recreate. Confirms: vector dimension is fixed per collection at creation time, not inferred per-upsert.

Final numbers:

| chunk_size | recall@5 | mrr   | ndcg@5 |
| :--------- | :------- | :---- | :----- |
| 256        | 0.750    | 0.522 | 0.579  |
| 512        | 0.750    | 0.459 | 0.531  |
| 1024       | 0.450    | 0.310 | 0.344  |


# ai-infra-portfolio

A document-intelligence API built to demonstrate **production AI infrastructure**, not just a RAG demo. Documents go in, grounded answers come out, and every request is traced, costed, evaluated, and gated in CI.

The LLM is deliberately small (1.5-3.8B, CPU-only, local). **The model is a placeholder; the infrastructure around it is the project.** Retrieval, tracing, evals, and the agent harness behave identically whether the backend is a local model or a frontier API.

> **Status:** actively developed. Tier 1 (RAG + observability + evals + serving) is complete. The agent layer (Pillar 5) is in progress. See [Status](#status) for exactly what exists today vs. what is planned.

---

## What this demonstrates

| Capability | Where to look |
|---|---|
| Production RAG: chunking, hybrid search (BM25 + dense + RRF), reranking | `[pillar-10.x]` commits |
| Observability: end-to-end tracing, cost per request, latency profiling, drift detection, alerting | `[pillar-2.x]` commits |
| Eval infrastructure: golden set, harness, CI gate that blocks regressions | `[pillar-11.x]` commits |
| Model serving & inference: vLLM vs SGLang, batching, quantization, KV cache, cost model | `[pillar-4.x]` commits |
| Agent layer: LangGraph, human-in-the-loop, reliable tool-calling | `[pillar-5.x]` commits (in progress) |

---

## Architecture

```mermaid
flowchart LR
    subgraph Ingest["Ingestion Service (FastAPI)"]
        D[Documents] --> C[Chunking pipeline]
        C --> E[Embedding service]
        E --> Q[(Qdrant)]
    end

    subgraph Query["Query Service (FastAPI)"]
        U[User query] --> R[Hybrid retrieval<br/>BM25 + dense + RRF]
        Q --> R
        R --> L[LLM call via Ollama]
        L --> A[Answer + sources]
    end

    subgraph Agent["Agent layer (in progress)"]
        G[LangGraph agent] --> T[Retrieval tool]
        T --> R
    end

    Query -. every request .-> LF[Langfuse Cloud<br/>traces, tokens, cost, latency]
    Agent -. every run .-> LF
    PG[(Postgres<br/>app data + eval results)] --- Query
    PG --- EV

    subgraph CI["GitHub Actions"]
        EV[Eval harness<br/>50 golden Q&A pairs] --> GATE{Recall@5 and judge score<br/>above threshold?}
        GATE -- no --> FAIL[Fail PR]
        GATE -- yes --> PASS[Merge allowed]
    end
```

> Retrieval is exposed to the agent as a tool today. In Session 5.8 it will be wrapped as an MCP server.

---

## Status

| Pillar | Topic | State |
|---|---|---|
| Pre-Work | FastAPI + Postgres + Docker Compose + Ollama | ✅ Done |
| 1 | Foundations: containerization, failure-point documentation | ✅ Done |
| 2 | Observability: Langfuse tracing, cost dashboard, latency profiling, drift detection, alert design | ✅ Done |
| 10 | Vector infrastructure: Qdrant, chunking experiments, hybrid search | ✅ Done |
| 11 | Eval infrastructure: harness, CI gate, behavioral tests, feedback loop | ✅ Done |
| 4 | Model serving: vLLM / SGLang benchmarks, quantization, embedding service, cost model | ✅ Done |
| 5 | Agent layer: LangGraph, tool-calling reliability, memory, MCP | 🔨 In progress (5.1-5.2 done, 5.3 current) |
| 12 | AI gateway (LiteLLM): routing, fallbacks, PII scrubbing | ⏳ Planned |
| 3 | Pipeline orchestration: Prefect, Kafka ingestion, SSE streaming, DVC | ⏳ Planned |
| 8 | Cost & performance: semantic cache, model router, cost attribution | ⏳ Planned |

Until Pillar 12 lands, LLM calls go directly to Ollama rather than through a gateway.

---

## Tech stack

| Layer | Choice |
|---|---|
| API | FastAPI, Pydantic, asyncpg |
| Vector store | Qdrant (see [design decisions](#design-decisions) for the pgvector comparison) |
| Relational | PostgreSQL (app data, eval results) |
| Local LLM | Ollama: `phi3:mini`, `gemma2:2b`, `qwen2.5:1.5b` |
| Agents | LangChain `create_agent` on the LangGraph runtime |
| Observability | Langfuse Cloud, Python SDK v4 |
| Evals / CI | Custom harness + GitHub Actions |
| GPU benchmarking | vLLM, SGLang on Kaggle T4 |
| Containers | Docker, Docker Compose |

---

## Constraints this was built under

This runs on a laptop, on purpose. Every design choice was made against real limits:

| Constraint | Value |
|---|---|
| CPU | AMD Ryzen 5 3500U (4 cores / 8 threads) |
| Usable RAM | **5.92 GB** (8 GB installed; the rest is reserved by the integrated GPU and BIOS) |
| GPU | Integrated Radeon Vega 8, no CUDA, no discrete VRAM |
| Cloud spend | $0 |
| GPU work | Kaggle free tier (T4) only |

**Memory discipline.** Every Compose service has explicit `mem_limit` / `memswap_limit` (api 512m, Qdrant 512m, Postgres 1g). Ollama runs natively, outside Docker. Model choice is tied to how many services are running: `qwen2.5:1.5b` (1.0 GB) when three or more services are up, `phi3:mini` (2.2 GB) only when model quality matters and RAM allows.

**Model roles.**

| Model | Role |
|---|---|
| `phi3:mini` | Default generation model for RAG, observability, and eval labs. No native tool calling in Ollama. |
| `gemma2:2b` | Second model for A/B tests and comparison runs |
| `qwen2.5:1.5b` | Agent / tool-calling model, LLM-as-judge, low-RAM runs |

---

## Quickstart

> Fill in the exact commands from your repo. The structure below matches how the project is meant to be run.

**Prerequisites:** Docker, Python (pyenv), Ollama, a free [Langfuse Cloud](https://cloud.langfuse.com) project.

```bash
# 1. Clone
git clone https://github.com/<your-username>/ai-infra-portfolio.git
cd ai-infra-portfolio

# 2. Pull local models
ollama pull phi3:mini
ollama pull gemma2:2b
ollama pull qwen2.5:1.5b
ollama serve            # keep running in its own terminal

# 3. Configure environment
cp .env.example .env    # add Langfuse keys and DB credentials

# 4. Check RAM before starting
docker stats --no-stream

# 5. Start services
docker compose up -d

# 6. TODO: ingest sample documents
# 7. TODO: run a sample query (curl example)
```

**Environment variables:** `.env` must be loaded before any Langfuse client is constructed. FastAPI/uvicorn does not auto-load it, so the app calls `load_dotenv()` first. To confirm vars loaded:

```bash
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print(bool(os.getenv('LANGFUSE_PUBLIC_KEY')))"
```

---

## Results

> Fill in with **your own measured numbers**. Leave a row out rather than guessing.

### Retrieval quality

| Experiment | Result |
|---|---|
| Dense-only vs hybrid (BM25 + dense + RRF), Recall@5 | TODO |
| Chunk size 256 / 512 / 1024 tokens vs Recall@5 | TODO |
| HNSW vs IVF recall@10 at different `ef` values | TODO |

### Evals

| Metric | Value |
|---|---|
| Golden set size | 50 Q&A pairs |
| Recall@5 | TODO |
| LLM-as-judge score | TODO |
| CI gate threshold | TODO |

### Serving (Kaggle T4)

| Experiment | Result |
|---|---|
| vLLM vs SGLang throughput at varying concurrency | TODO |
| Static vs continuous batching | TODO |
| GPTQ quantization: quality delta vs latency delta | TODO |
| Self-host vs API break-even (daily token volume) | TODO |

### Latency (local, CPU)

Local generation runs at roughly 15-30 tokens/second on this hardware; requests take 10-30 seconds. This is expected for a CPU-only laptop, and all throughput benchmarking happens on Kaggle instead.

| Stage | P50 | P95 |
|---|---|---|
| Retrieval | TODO | TODO |
| Generation | TODO | TODO |
| End-to-end | TODO | TODO |

---

## Observability

Every request produces a Langfuse trace with a span per pipeline stage (retrieve, rerank, generate), token counts, cost attribution, and latency. Agent runs are traced through the Langfuse LangChain callback handler.

What is monitored and why: see `docs/alerting.md` (TODO: link) for the alert matrix covering latency, cost, error rate, and quality drift.

---

## Evals and CI gate

```
PR opened
   │
   ▼
GitHub Actions ──► eval harness ──► 50 golden Q&A pairs
                                        │
                        ┌───────────────┴───────────────┐
                        ▼                               ▼
                  Recall@5 score                LLM-as-judge score
                        └───────────────┬───────────────┘
                                        ▼
                         below threshold?  ──yes──► PR blocked
                                        │
                                        no
                                        ▼
                                  merge allowed
```

Beyond the golden set, a behavioral test suite checks behavior rather than exact output: hallucination guard, citation accuracy, and refusal on out-of-scope questions. A thumbs up/down endpoint feeds high-signal examples back into the golden set.

---

## Agent layer (in progress)

An agent built on top of the RAG pipeline, using LangGraph.

**Done:** orchestration pattern comparison (ReAct vs plan-and-execute), a LangGraph agent with a human-in-the-loop checkpoint.

**In progress:** reliable tool-calling (retries, timeouts, idempotency, fallbacks).

**Planned:** episodic memory in Postgres, checkpoint/resume, retrieval tool exposed as an MCP server.

**Design stance:** the agent runs on a 1.5B model that picks wrong tools, emits malformed arguments, and loops. That is useful, not a defect. The work is hardening the harness so it survives those failures, not coaxing the model into behaving.

### Agent failure log

Each failure hit while building the agent, what it looked like, and the fix. Add an entry every time something breaks.

| # | Failure | Symptom | Fix in harness |
|---|---|---|---|
| 1 | TODO (e.g. flaky tool returning errors ~30% of calls) | TODO | TODO (e.g. bounded retry with backoff, then fallback) |
| 2 | TODO | TODO | TODO |

---

## Design decisions

**Qdrant vs pgvector.** Qdrant is used here for a dedicated vector index with tunable HNSW parameters and payload filtering. For a team already running Postgres at modest scale, pgvector is often the better default: one fewer service to operate, transactional consistency with relational data. Dedicated vector DBs earn their place when scale, filtering complexity, or index tuning outgrow what Postgres handles comfortably.

**Hybrid retrieval.** Dense-only retrieval misses exact-match queries (IDs, names, rare terms). BM25 + dense fused with Reciprocal Rank Fusion covers both. Measured against the dense-only baseline in the [results](#retrieval-quality).

**Langfuse Cloud, not self-hosted.** Self-hosted Langfuse v3/v4 requires a multi-container stack that does not fit in 5.92 GB alongside the rest of the system. Cloud free tier removes the problem.

**vLLM and SGLang for serving experiments.** Both are production-standard engines. Hugging Face archived TGI in March 2026 and recommends them for new deployments. Triton is understood conceptually (multi-model serving) but not a lab target here.

**Local-first, $0 budget.** Everything runs locally except GPU benchmarks, which run on Kaggle. No paid cloud services anywhere in the project.

---

## Roadmap

- [ ] **Pillar 5:** reliable tool-calling, episodic memory, checkpointing, guardrails, MCP server for retrieval
- [ ] **Pillar 12:** LiteLLM gateway, multi-provider fallback chains, PII scrubbing middleware
- [ ] **Pillar 3:** Prefect ingestion DAG, Kafka consumer for near-real-time indexing, SSE streaming, DVC data versioning
- [ ] **Pillar 8:** semantic cache, cost attribution, complexity-based model router

---

## Repository layout

> Replace with the output of `tree -L 2 -I '__pycache__|node_modules|.git'` from your repo root.

```
TODO
```

---

## Commit convention

```
[pillar-X.Y] short description of what was built
```

Example: `[pillar-5.3] add retry + fallback harness for flaky tools`

---

## Author

**<Your Name>** · [LinkedIn](https://linkedin.com/in/<handle>) · [GitHub](https://github.com/<your-username>)

Background: frontend/MERN engineer moving into AI infrastructure. Looking for Junior AI Infrastructure / MLOps roles in India.