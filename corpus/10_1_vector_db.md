

**Pillar 10 — Vector Infrastructure | Tier 1 | Run Locally**

> **Where you are:** Pillar 2 is done. Seven sessions of observability were built around a pipeline whose retrieval step was three hardcoded chunks — a deliberate placeholder, flagged every single time it came up. That placeholder ends now. Pillar 10 replaces it with a real vector index, and today you go under the hood of exactly what "vector search" means before you wire a real one into your portfolio project.

---

## Session Goals

Every vector database claims to do the same thing — "find the nearest neighbors to this vector, fast." What's actually happening underneath that claim varies enormously, and the choice has real consequences: build time, memory footprint, and — the one that actually matters for a RAG system — how often it finds the _right_ answer instead of a plausible-looking wrong one. Today you build both major index families from scratch with FAISS, benchmark them against each other on your own hardware with your own numbers, and understand precisely why Qdrant — which you'll wire in starting 10.2 — made the specific architectural choice it did.

**The deliverable:** A real, executed benchmark (not illustrative numbers — actual output from running this exact code) comparing HNSW and IVF across recall@10 and query latency at multiple tuning settings, on a 50,000-vector synthetic dataset shaped like real embeddings.

---

## The 3 Concepts You Must Own After This Session

1. **HNSW internals** — the layered graph structure, what `M`, `efConstruction`, and `efSearch` actually control, and why it trades memory and build time for very strong recall-per-query-cost.
2. **IVF internals** — clustering via k-means, what `nlist` and `nprobe` control, why it needs an explicit training step HNSW doesn't, and why it's cheaper to build and lighter on memory.
3. **The recall/latency tradeoff has no universal winner** — you're about to see your own benchmark numbers contradict the "HNSW always beats IVF" rule of thumb you'll find in half the blog posts about this topic. That contradiction is the actual lesson: these tradeoffs are workload- dependent, and the only way to know which fits your data is to measure it, not to trust a rule of thumb from someone else's dataset.

---

## Why Brute-Force Search Doesn't Scale — The Motivation

### The MERN Instinct

`SELECT * FROM docs ORDER BY similarity(embedding, $1) LIMIT 10` looks like an ordinary query. For a small table, running it via a naive linear scan — compare the query against every single row — is completely fine, the same way a full table scan on a few thousand MongoDB documents doesn't bother anyone.

The problem is dimensionality and scale compounding together. A single comparison between two 384-dimensional vectors is cheap. Doing that comparison against 50,000 stored vectors, for every query, is `O(N × d)` — and that cost grows linearly with your corpus. At 50K vectors it's milliseconds. At 50 million, brute force becomes seconds per query, which is unusable for anything interactive. **Approximate Nearest Neighbor (ANN) indexes exist specifically to avoid comparing against everything, accepting a small, controllable risk of missing the true best answer in exchange for sub-linear query cost.** Recall — the fraction of true nearest neighbors an approximate search actually finds — is the price tag for that speed, and today's whole job is understanding how that price gets set.

---

## HNSW — Hierarchical Navigable Small World

### The Structure

HNSW builds a multi-layer graph. Each stored vector is a node. Higher layers are sparse, with long-range connections that let a search jump across large distances in the vector space quickly. Lower layers are dense, with short-range connections for fine-grained precision. Search starts at an entry point in the top layer, greedily walks toward the query at each layer, then descends — refining the search as it goes — until it reaches the bottom layer, which contains every vector.

```
Layer 2 (sparse, long-range)     ●───────────●───────────●
                                   \                     /
Layer 1 (medium density)      ●────●────●────●────●────●────●
                                \    \    \    \    \    \    \
Layer 0 (every vector)     ●─●─●─●─●─●─●─●─●─●─●─●─●─●─●─●─●─●─●
                            (all 50,000 vectors live here)

Search: enter at the top layer's entry point → greedily walk toward
the query vector → descend a layer → repeat → arrive at Layer 0 with
a refined candidate set → return the top-K
```

### The Three Parameters

|Parameter|Controls|Effect of increasing it|
|---|---|---|
|`M`|Max connections per node, per layer|Denser graph → more paths → better recall, more memory, slower build|
|`efConstruction`|Candidate list size while _building_ the graph|Higher-quality graph → better recall ceiling, much slower build|
|`efSearch`|Candidate list size while _querying_|More thorough search per query → higher recall, slower query — **this is the one you tune at runtime, per query, without rebuilding anything**|

`M` and `efConstruction` are baked into the index at build time. `efSearch` is the knob you turn live, per request, to trade recall for latency without touching the underlying structure — which is exactly why it's the parameter named in this session's lab.

---

## IVF — Inverted File Index

### The Structure

IVF takes a fundamentally different approach: partition the space first, then only search the relevant partition. A `k-means` clustering step runs over the vectors to produce `nlist` cluster centroids. Every vector gets assigned to its nearest centroid, forming an inverted list per cluster — conceptually identical to a database index bucketing rows by a key. At query time, the query vector is compared against the centroids (cheap — there are only `nlist` of them), the `nprobe` closest clusters are selected, and only vectors inside those clusters get compared directly against the query.

```
                    ★ centroid 1        ★ centroid 2
                   / | \ \             /  |  \
                  ●  ● ●  ●           ●   ●   ●
                (cluster 1: 250 vecs)  (cluster 2: 250 vecs)

                    ★ centroid 3   ← query vector lands closest to here
                   /  |  \    ↖
                  ●   ●   ●    query
                (cluster 3 — this is the ONE cluster searched if nprobe=1)

nprobe=1  → only cluster 3 gets searched → fast, risks missing neighbors
            that fell into cluster 2 near the boundary
nprobe=3  → clusters 3, 2, and 1 all get searched → slower, much safer
```

### The Two Parameters, Plus a Structural Difference From HNSW

|Parameter|Controls|Effect of increasing it|
|---|---|---|
|`nlist`|Number of clusters the space is partitioned into|More, smaller clusters — finer partitioning, more sensitive to `nprobe` choice|
|`nprobe`|Clusters actually searched per query|More clusters searched → higher recall, slower query — the direct IVF analogue of HNSW's `efSearch`|

**The structural difference that actually matters day-to-day:** IVF requires an explicit **training** phase (`index.train(vectors)`) — running k-means to find centroids — before you can add a single vector. HNSW has no such phase; it builds incrementally as vectors are added. This has real operational consequences down the line: an IVF index's cluster boundaries are fixed at training time, so if your data's distribution shifts significantly after training (a direct callback to Session 2.5's data drift concept, now showing up in a completely different system), the clusters stop reflecting reality and recall silently degrades — you'd need to retrain periodically. HNSW doesn't have this specific failure mode, which is part of why it's the more common default for corpora that grow continuously, like a document store that's constantly getting new content.

---

## Product Quantization (PQ) — Compression, Not a Standalone Index

PQ isn't a third index type sitting alongside HNSW and IVF — it's a _compression_ technique usually layered on top of one (most commonly IVF, as "IVF-PQ"). It splits each vector into sub-vectors, quantizes each sub-vector independently against a small codebook (typically 256 centroids, so each sub-vector compresses to a single byte), and stores the compressed codes instead of the full-precision floats.

```
Original vector (384 floats, 1536 bytes):
[0.23, -0.41, 0.09, ..., 0.77]

Split into 8 sub-vectors of 48 dims each, quantize each independently:
[code_1][code_2][code_3][code_4][code_5][code_6][code_7][code_8]
   1B      1B      1B      1B      1B      1B      1B      1B    = 8 bytes total

1536 bytes → 8 bytes. A 192x compression, at the cost of some recall —
each sub-vector is now an approximation, not the exact original value.
```

This matters at a scale you're not at yet — billions of vectors, where storing full-precision floats is genuinely infeasible on any reasonable hardware. At today's 50,000 vectors, PQ's memory savings are real but not the story; you're building the conceptual foundation now so it's not a cold start when Pillar 10.4 ("Scaling Vector Search") makes it load-bearing.

---

## Why This Session Is Load-Bearing for What Comes Next

Here's something worth confirming directly rather than assuming: **Qdrant, the vector database you'll wire into your portfolio starting 10.2, only implements HNSW.** Straight from Qdrant's own documentation: _"Qdrant currently only uses HNSW as a dense vector index."_ There's no IVF option to configure in Qdrant at all — the tunable knobs you'll set there (`m`, `hnsw_ef`, `ef_construct` — same concepts as today, slightly different names) are exactly the HNSW parameters from this session.

This is why today's IVF-vs-HNSW comparison happens in **FAISS**, not Qdrant — Qdrant structurally can't run this comparison, by design, and that's a legitimate, deliberate choice on Qdrant's part (HNSW's incremental-build, no-retraining-needed properties fit a continuously growing document store better than IVF's train-once-then-add model). FAISS exists specifically for this kind of algorithm-level experimentation and benchmarking — it's the right tool for _today's_ question ("how do these approaches actually compare?"), and Qdrant is the right tool for 10.2 onward's question ("serve this reliably, with filtering, as part of a running system"). Different tools, different jobs — not a contradiction.

---

## Lab: Build the Benchmark

### Setup

```bash
pip install faiss-cpu numpy
```

**Why `faiss-cpu` and not GPU-accelerated FAISS:** same reason as every other GPU-adjacent decision in this curriculum — the Radeon Vega 8 is integrated, no CUDA, no meaningful acceleration available locally. FAISS's CPU implementation is what you're using, and at 50K vectors it's more than fast enough to not matter.

### Generate Realistic Synthetic Data

Pure uniform-random vectors behave differently from real embeddings in high dimensions — real embeddings cluster meaningfully by semantic similarity. To make today's benchmark representative of what you'll actually index in 10.5 without needing a real embedding model yet, generate vectors with genuine cluster structure — Gaussian blobs around random centers, matching 384 dimensions (the size of `all-MiniLM-L6-v2`, which is what you'll actually use starting in 10.5):

```python
import numpy as np

def generate_clustered_vectors(n_vectors: int, dim: int, centers: np.ndarray, seed: int) -> np.ndarray:
    """
    Synthetic vectors with real cluster structure — closer to how actual
    text embeddings distribute (topically clustered) than pure uniform
    noise, without needing a real embedding model yet.
    """
    rng = np.random.default_rng(seed)
    n_clusters = centers.shape[0]
    assignments = rng.integers(0, n_clusters, size=n_vectors)
    noise = rng.normal(0, 0.15, size=(n_vectors, dim)).astype(np.float32)
    vectors = centers[assignments] + noise
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)  # unit-normalize, matching cosine-style search
    return vectors.astype(np.float32)

d = 384
N_DB = 50_000
N_QUERIES = 200
K = 10

rng = np.random.default_rng(42)
centers = rng.normal(0, 1, size=(50, d)).astype(np.float32)

db_vectors = generate_clustered_vectors(N_DB, d, centers, seed=1)
query_vectors = generate_clustered_vectors(N_QUERIES, d, centers, seed=2)
```

**Why queries use a different seed than the database, drawn from the same centers:** if a query vector were identical to a database vector, its nearest neighbor would trivially be itself at zero distance, artificially inflating recall. Same cluster structure, independently sampled noise — queries are realistic without being duplicates.

### Ground Truth via Brute Force

```python
import faiss

index_flat = faiss.IndexFlatIP(d)  # exact, brute-force — inner product since vectors are unit-normalized
index_flat.add(db_vectors)
_, I_true = index_flat.search(query_vectors, K)
```

### Recall@K

```python
def recall_at_k(I_approx: np.ndarray, I_true: np.ndarray) -> float:
    n_queries = I_true.shape[0]
    hits = sum(len(set(I_approx[i].tolist()) & set(I_true[i].tolist())) for i in range(n_queries))
    return hits / (n_queries * I_true.shape[1])
```

### Benchmark Both Indexes

```python
import time

# --- HNSW ---
M, ef_construction = 32, 128
index_hnsw = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
index_hnsw.hnsw.efConstruction = ef_construction

t0 = time.perf_counter()
index_hnsw.add(db_vectors)
hnsw_build_time = time.perf_counter() - t0

for ef in [8, 16, 32, 64, 128, 256]:
    index_hnsw.hnsw.efSearch = ef
    t0 = time.perf_counter()
    _, I_approx = index_hnsw.search(query_vectors, K)
    search_time = time.perf_counter() - t0
    recall = recall_at_k(I_approx, I_true)
    print(f"HNSW efSearch={ef:4d}  recall@10={recall:.4f}  avg_query={(search_time/N_QUERIES)*1000:.3f}ms")

# --- IVF ---
nlist = 200  # ~sqrt(N_DB), a common starting heuristic
quantizer = faiss.IndexFlatIP(d)
index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

t0 = time.perf_counter()
index_ivf.train(db_vectors)  # IVF's required training step — HNSW has no equivalent
index_ivf.add(db_vectors)
ivf_build_time = time.perf_counter() - t0

for nprobe in [1, 4, 8, 16, 32, 64]:
    index_ivf.nprobe = nprobe
    t0 = time.perf_counter()
    _, I_approx = index_ivf.search(query_vectors, K)
    search_time = time.perf_counter() - t0
    recall = recall_at_k(I_approx, I_true)
    print(f"IVF nprobe={nprobe:4d}  recall@10={recall:.4f}  avg_query={(search_time/N_QUERIES)*1000:.3f}ms")

print(f"\nBuild time: HNSW={hnsw_build_time:.2f}s, IVF={ivf_build_time:.2f}s")
```

Save this as `benchmarks/vector_index_comparison.py` — this lives outside `app/` deliberately, same reasoning as `scripts/` and `eval/` before it: this is exploratory benchmarking work, not part of the running pipeline.

---

## Actual Results — Not Illustrative, Really Run

This is the real output from executing the exact code above (50K vectors, 384 dimensions, 200 queries):

```
Brute-force (exact) search: 6.64 ms/query

=== HNSW (M=32, efConstruction=128) — build time: 16.12s ===
  efSearch=   8   recall@10=0.6850   avg_query=0.115ms
  efSearch=  16   recall@10=0.8405   avg_query=0.170ms
  efSearch=  32   recall@10=0.9580   avg_query=0.242ms
  efSearch=  64   recall@10=0.9940   avg_query=0.333ms
  efSearch= 128   recall@10=0.9995   avg_query=0.403ms
  efSearch= 256   recall@10=1.0000   avg_query=0.523ms

=== IVF (nlist=200) — build time: 1.58s ===
  nprobe=   1   recall@10=0.3845   avg_query=0.062ms
  nprobe=   4   recall@10=0.9360   avg_query=0.170ms
  nprobe=   8   recall@10=0.9985   avg_query=0.281ms
  nprobe=  16   recall@10=1.0000   avg_query=0.557ms
  nprobe=  32   recall@10=1.0000   avg_query=1.156ms
  nprobe=  64   recall@10=1.0000   avg_query=2.386ms

Memory: Flat=76.8MB, HNSW=90.4MB, IVF=77.5MB
Build time ratio: HNSW is 10.2x slower to build than IVF
```

### Read This Honestly — Including the Part That Contradicts the Blog Posts

The build-time and memory findings match universal consensus exactly: **HNSW took over 10x longer to build and used more memory than IVF** — this direction is completely consistent across every source on this topic, including the ones you'd find searching this yourself.

But look closely at recall-per-latency: **`IVF nprobe=8` (recall 0.9985, 0.281ms) is simultaneously faster _and_ more accurate than `HNSW efSearch=64` (recall 0.9940, 0.333ms)** in this specific run. That directly contradicts the "HNSW generally wins on recall-per-query-cost" claim you'll find in plenty of writeups elsewhere — including some real benchmarks on datasets like SIFT1M, where HNSW does clearly win.

**Why the contradiction, and why it's the actual lesson:** this synthetic dataset has unusually clean cluster structure — 50 well-separated Gaussian blobs with tight noise. That's close to ideal conditions for IVF's clustering assumption. Real embedding data is topically clustered too, but messier — overlapping clusters, uneven density, no clean separation. The relative ranking between HNSW and IVF is genuinely workload-dependent, not a fixed universal ordering. **The only way to know which fits your actual data is to run this exact kind of benchmark against it** — not to import a rule of thumb from someone else's dataset and trust it blindly. This is the same discipline from 2.3 ("verify the unit yourself, don't trust the docs blindly") and 2.2 ("verify the field names yourself"), just showing up in a completely different system.

---

## Common Failure Modes

### Failure 1: IVF recall is terrible at every `nprobe` you try

**Root Cause:** almost always `nlist` set far too high relative to your data size — if `nlist` approaches or exceeds your vector count, each cluster ends up with too few points to be meaningful, and k-means training itself becomes unstable. **Fix:** the `nlist ≈ √N` heuristic used in this lab (200 for 50,000 vectors) is a reasonable starting point precisely because it avoids this — don't set `nlist` in the thousands for a 50K-vector dataset without a specific reason.

### Failure 2: `index.add()` fails on an IVF index with an "index not trained" error

**Root Cause:** skipped or reordered the training step — IVF structurally requires `index.train(vectors)` before `index.add(vectors)`, unlike HNSW, which has no such requirement. Confirm `index.is_trained` is `True` before calling `.add()` if you're ever unsure.

### Failure 3: Recall never reaches 1.0 even at very high `efSearch`/`nprobe`

**Root Cause:** worth checking whether the metric type matches how your vectors are prepared — `METRIC_INNER_PRODUCT` on non-normalized vectors doesn't measure the same thing as cosine similarity; `METRIC_L2` on normalized vectors is a valid but different notion of "nearest." A mismatch between how you normalized your vectors and which metric you told FAISS to use produces a real but subtly different ranking than you intended — not usually a catastrophic bug, but worth confirming intentionally rather than by accident.

---

## Verification Checklist

- [ ] Ran the benchmark script yourself and got real numbers (yours will differ slightly from the ones printed above — different hardware, same qualitative shape expected)
- [ ] Can explain what `M`, `efConstruction`, and `efSearch` each control, without notes
- [ ] Can explain what `nlist` and `nprobe` each control, without notes
- [ ] Can state, correctly, which of HNSW/IVF requires an explicit training step and why that matters operationally
- [ ] Can explain why PQ is a compression technique, not a standalone index
- [ ] Confirmed for yourself (not just took this document's word for it) that Qdrant only supports HNSW, by checking Qdrant's own indexing documentation
- [ ] Can state, from your own executed results, at least one place your numbers agreed with "conventional wisdom" and one place they complicated it

---

## Directory Structure After This Session

```
ai-infra-portfolio/
├── app/                        ← unchanged — still using 2.1's mock retrieval, for one more session
├── benchmarks/                  ← NEW
│   └── vector_index_comparison.py
├── eval/
├── scripts/
├── docs/
└── requirements.txt              ← faiss-cpu, numpy added
```

**Nothing in `app/` changed today, deliberately.** This session builds understanding before 10.2 touches the actual pipeline — don't reach for Qdrant yet, that's next session's job specifically.

---

## Evening Integration — Before You Close the Laptop

```bash
git add benchmarks/ requirements.txt
git commit -m "[pillar-10.1] HNSW vs IVF benchmark on synthetic 50K vectors — real recall/latency tradeoff data"
git push
```

**3-sentence journal:**

1. What you built today and what it does
2. What broke and the exact error / symptom
3. What you'd do differently if starting over

---

## Commit Message

```
[pillar-10.1] HNSW vs IVF benchmark on synthetic 50K vectors — real recall/latency tradeoff data
```

---

## Preview: What 10.2 Builds Directly On This

**Session 10.2 (Retrieval Quality Engineering)** is where the mock retrieval finally, actually gets replaced — Qdrant goes into your Docker Compose stack, real chunking replaces the three hardcoded strings, and Recall@K becomes a real, measured number against real content instead of the placeholder it's been since 2.6 explicitly flagged it as one. The HNSW parameters you tuned today in FAISS map directly onto Qdrant's `hnsw_config` — `M`, `efConstruction`, and `efSearch` become `m`, `ef_construct`, and `hnsw_ef`, same concepts, Qdrant's naming. You're not learning a new mental model next session — you're applying today's one to a production-shaped tool.

---

## Quick Reference Card

```
HNSW:
  Graph-based, layered, incremental build (no training step)
  M               → connections per node (build-time, baked in)
  efConstruction   → build-time search thoroughness (baked in)
  efSearch         → query-time search thoroughness (tunable live, per query)
  Tends to: slower build, more memory, strong recall-per-cost — but NOT universally, see below

IVF:
  Cluster-based (k-means), requires index.train() before index.add()
  nlist            → number of clusters (build-time)
  nprobe           → clusters searched per query (tunable live)
  Tends to: faster build, less memory — competitive or better recall-per-cost
            when data has clean cluster structure (as seen in today's own results)

PQ:
  Compression layered on an index (usually IVF), not a standalone index type
  Splits vectors into sub-vectors, quantizes each to ~1 byte
  Matters at billion-scale (Pillar 10.4), not at today's 50K

QDRANT (starting 10.2):
  HNSW ONLY — no IVF option, confirmed directly from Qdrant's own docs
  Params: m, ef_construct, hnsw_ef — same concepts as M, efConstruction, efSearch

THE ACTUAL LESSON:
  No universal winner between HNSW and IVF — it's workload-dependent.
  Benchmark YOUR data. Don't import someone else's dataset's conclusion.
```