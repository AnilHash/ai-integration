**Pillar 10 — Vector Infrastructure | Tier 1 | Run Locally**

> **Where you are:** 10.1–10.4 are done — HNSW/IVF tradeoffs, real Recall@5/MRR/NDCG@5 numbers, hybrid+reranked retrieval live in `retrieve_documents()`, and a scaling benchmark that told you honestly what breaks at 10M vectors. Everything so far assumed your document set is static: load once, query forever. Today it isn't anymore. This is the last session of Pillar 10 — after today, check it off in your tier tracker and move to Pillar 11.

---

## Session Goals

Your real `docs_512` collection has only ever been built one way: `scripts/load_chunks.py` drops the whole collection and rebuilds it from scratch. That's fine once. It's not fine when a document changes next week. Today you build the three operations a real pipeline needs — add a new document, update a changed one, delete a removed one — without ever nuking the collection again. Along the way you'll fix a real design flaw carried since 10.2, watch what "index drift" actually looks like on your own collection, and safely migrate to a new embedding model with zero downtime using the technique real teams use for exactly this.

**The deliverable:** Stable, content-independent chunk IDs; working add/update/delete scripts that touch only the points they need to; a documented before/after of a real drift observation on `docs_512`; and an alias-based embedding-model migration you can describe accurately in an interview.

---

## The 3 Concepts You Must Own After This Session

1. **Incremental indexing requires IDs that don't depend on what else is in the collection.** Your ID scheme since 10.2 (`point_id = 0, 1, 2, ...` in doc-iteration order) works great for a one-time bulk load and breaks the moment you add a document — insert anything before the alphabetically-last doc and every ID after it shifts, silently corrupting the BM25 alignment from 10.3. The fix is a _stable_ ID: deterministic from `(source_doc, chunk_index)`, independent of what else exists.
2. **Deletion in Qdrant is soft — and that's intentional, not a bug to route around.** Qdrant marks deleted points with a bitmask rather than physically removing them immediately, because touching disk on every delete is expensive. A background Vacuum Optimizer does the real cleanup later, once a segment's deleted fraction crosses a threshold. `points_count` and `vectors_count` are explicitly documented as approximate for exactly this reason — don't write logic that depends on them being exact immediately after a write.
3. **You cannot mix vectors from two different embedding models, and Qdrant will not stop you from trying.** Two models can output the same dimensionality (384 and 384) while placing semantically similar text in completely different regions of that space. Querying a new-model collection with an old-model query vector returns results — no dimension error, no warning — they're just meaningless. The only safe migration path is a parallel collection and an atomic alias swap, never an in-place vector overwrite.

---

## Part A: Prove the 10.2 ID Scheme Is Actually Fragile

Before fixing it, see the break. Your current `load_chunks.py` assigns `point_id` by iterating `sorted(CORPUS_DIR.glob("*.md"))` and counting up from 0. Add a new file that sorts _before_ your existing ones alphabetically:

```bash
echo "# 0_test\n\nThis is a test document for Part A." > corpus/0_test_doc.md
```

If you reran `load_chunks.py` right now, `0_test_doc.md`'s chunks would claim IDs `0, 1, ...`, and every chunk from every other document would shift down by however many chunks the new file added — silently reassigning IDs that your BM25 corpus-position mapping from 10.3 depends on matching exactly. **Don't actually rerun it** — delete the test file and keep this as the motivating example, not a real change:

```bash
rm corpus/0_test_doc.md
```

This is why today's first real task is replacing position-based IDs with content-independent ones, before touching anything else.

---

## Part B: Stable Chunk IDs

```python
# chunking/stable_ids.py
import uuid

# Fixed, arbitrary namespace UUID -- must NEVER change once real points exist.
# Changing it recomputes every chunk's ID differently, which defeats the
# entire point of "stable."
NAMESPACE = uuid.UUID("7c9e6679-7425-40de-944b-e07fc1f90ae7")

def stable_chunk_id(source_doc: str, chunk_index: int) -> str:
    """Deterministic UUID from (doc, position-within-doc). Same doc, same
    chunker, same chunk_index -> always the same ID, regardless of what
    else exists in the collection. Qdrant accepts UUID strings as point IDs."""
    return str(uuid.uuid5(NAMESPACE, f"{source_doc}::{chunk_index}"))
```

This ID is a function of `(source_doc, chunk_index)` only — never of the chunk's text content. That's a deliberate choice, not an oversight: it means re-processing an _unchanged_ document is idempotent (same IDs, same final state) while still going through a real delete-then-reinsert cycle each time you explicitly call update — which matters for Part F.

---

## Part C: Incremental Add — Touch Only the New Document

```python
# scripts/add_document.py
import sys
from pathlib import Path
from qdrant_client import QdrantClient, models
from app.embeddings import get_embedder
from chunking.chunker import chunk_text
from chunking.stable_ids import stable_chunk_id

CHUNK_SIZE = 512
client = QdrantClient(url="http://localhost:6333")

def add_document(path: Path):
    doc_id = path.stem
    text = path.read_text(encoding="utf-8")
    embedder = get_embedder()

    points = []
    for i, chunk in enumerate(chunk_text(text, CHUNK_SIZE)):
        points.append(models.PointStruct(
            id=stable_chunk_id(doc_id, i),
            vector=embedder.encode(chunk).tolist(),
            payload={"source_doc": doc_id, "text": chunk, "chunk_size": CHUNK_SIZE},
        ))

    client.upsert(collection_name="docs_512", points=points)
    print(f"Added {len(points)} chunks for '{doc_id}' -- no existing points touched")

if __name__ == "__main__":
    add_document(Path(sys.argv[1]))
```

Copy this very session's notes in as your first real test — self-referential, but it's real content you understand:

```bash
cp Session_10_5*.md corpus/10_5_embedding_ops.md
python scripts/add_document.py corpus/10_5_embedding_ops.md
```

Verify nothing else moved:

```bash
python -c "
from qdrant_client import QdrantClient
c = QdrantClient(url='http://localhost:6333')
print(c.count('docs_512').count)
"
```

The count should go up by exactly the new document's chunk count — every prior chunk's ID is untouched, because none of them were ever a function of document count or order.

---

## Part D: Payload Index — Make Filtered Delete/Update Actually Fast

You're about to delete and update by `source_doc` filter repeatedly. Without an index on that field, every filtered operation falls back to a full collection scan — harmless at ~100 points, genuinely costly at the scale you benchmarked in 10.4. Set it up once, now:

```python
from qdrant_client import QdrantClient, models
client = QdrantClient(url="http://localhost:6333")
client.create_payload_index(
    collection_name="docs_512",
    field_name="source_doc",
    field_schema=models.PayloadSchemaType.KEYWORD,
)
```

---

## Part E: Update and Delete — Filter-Based, Not ID-Based

```python
# scripts/update_document.py -- delete this doc's old chunks, then re-add fresh ones
import sys
from pathlib import Path
from qdrant_client import QdrantClient, models
from scripts.add_document import add_document

client = QdrantClient(url="http://localhost:6333")

def update_document(path: Path):
    doc_id = path.stem
    client.delete(
        collection_name="docs_512",
        points_selector=models.FilterSelector(
            filter=models.Filter(must=[
                models.FieldCondition(key="source_doc", match=models.MatchValue(value=doc_id))
            ])
        ),
    )
    add_document(path)  # fresh chunk + embed + upsert -- stable IDs, correct count either way
    print(f"Updated '{doc_id}': old chunks removed, new chunks added")

if __name__ == "__main__":
    update_document(Path(sys.argv[1]))
```

```python
# scripts/delete_document.py -- deletion only, no re-embedding
import sys
from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333")

def delete_document(doc_id: str):
    client.delete(
        collection_name="docs_512",
        points_selector=models.FilterSelector(
            filter=models.Filter(must=[
                models.FieldCondition(key="source_doc", match=models.MatchValue(value=doc_id))
            ])
        ),
    )
    print(f"Deleted all chunks for '{doc_id}'")

if __name__ == "__main__":
    delete_document(sys.argv[1])
```

**Why delete-then-reinsert instead of trying to diff old vs. new chunks:** if a document shrinks, some old chunk positions simply cease to exist; if it grows, new positions appear. Trying to patch individual chunks in place means tracking which positions are stale — filter-delete-by-doc then full re-add is simpler, correct by construction, and cheap at this scale.

```bash
python scripts/update_document.py corpus/10_5_embedding_ops.md   # re-run on the file you just added
python scripts/delete_document.py 10_5_embedding_ops              # clean up your test doc
```

---

## Part F: Watch Two Real (and Different) Things Happen

Run this before and after a few update cycles:

```python
from qdrant_client import QdrantClient
client = QdrantClient(url="http://localhost:6333")
info = client.get_collection("docs_512")
print("status:", info.status, "| points:", info.points_count,
      "| vectors:", info.vectors_count, "| indexed:", info.indexed_vectors_count)
```

**Thing 1 — `indexed_vectors_count` is probably near 0, and that's correct.** Qdrant only builds an HNSW segment once a collection's _unindexed_ vector data crosses `indexing_threshold` (20,000 kB by default). At ~100 chunks × 384 floats, `docs_512` is nowhere close — Qdrant is deliberately doing full-scan brute-force search instead of maintaining a graph, because at this size brute force is _faster_ than HNSW overhead. This is the same tradeoff from 10.1's HNSW-vs-exact discussion, just showing up from the other direction: below a real threshold, the "advanced" index isn't just unnecessary, it's the wrong choice.

**Thing 2 — `points_count` and `vectors_count` are approximate, and can look temporarily inconsistent after churn.** Run a few update cycles on a real document:

```bash
for i in 1 2 3; do
  python scripts/update_document.py corpus/10_1_vector_db.md
done
```

Check `points_count` right after each cycle. Because deletes are soft (bitmask) and cleanup batches in the background, these numbers are officially documented as not-exact by design — they reflect internal storage state, not a live, precise count at any instant. If you need an exact number for anything real, use `client.count("docs_512").count`, not `get_collection().points_count`. **This gap between "what the collection info reports" and "what's actually true" is index drift, made concrete** — not a hypothetical, something you just watched happen on your own collection.

---

## Part G: Zero-Downtime Embedding Model Migration

### Step 1 — Wire up an alias, retroactively

```python
# scripts/create_alias.py -- run once
from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333")
client.update_collection_aliases(
    change_aliases_operations=[
        models.CreateAliasOperation(
            create_alias=models.CreateAlias(collection_name="docs_512", alias_name="docs_active")
        )
    ]
)
print("alias 'docs_active' -> 'docs_512'")
```

Update `app/rag_pipeline.py` to query `"docs_active"` instead of the hardcoded `"docs_512"` everywhere it currently appears — a one-line-per-call-site change, done once, that makes every future migration a background operation instead of a code deploy.

### Step 2 — Build a parallel collection with a different model

```python
# scripts/migrate_embedding_model.py
from pathlib import Path
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from chunking.chunker import chunk_text
from chunking.stable_ids import stable_chunk_id

CORPUS_DIR = Path("corpus")
CHUNK_SIZE = 512
NEW_COLLECTION = "docs_512_v2"
NEW_MODEL = "sentence-transformers/all-MiniLM-L12-v2"  # stand-in "new embedding model" -- deeper network, same 384 dim

client = QdrantClient(url="http://localhost:6333")
new_embedder = SentenceTransformer(NEW_MODEL)

def build_new_collection():
    if client.collection_exists(NEW_COLLECTION):
        client.delete_collection(NEW_COLLECTION)
    client.create_collection(
        collection_name=NEW_COLLECTION,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
        hnsw_config=models.HnswConfigDiff(m=32, ef_construct=200),
    )
    points = []
    for path in sorted(CORPUS_DIR.glob("*.md")):
        doc_id = path.stem
        for i, chunk in enumerate(chunk_text(path.read_text(encoding="utf-8"), CHUNK_SIZE)):
            points.append(models.PointStruct(
                id=stable_chunk_id(doc_id, i),   # SAME ID scheme -- same logical chunk, new vector space
                vector=new_embedder.encode(chunk).tolist(),
                payload={"source_doc": doc_id, "text": chunk, "chunk_size": CHUNK_SIZE},
            ))
    client.upsert(collection_name=NEW_COLLECTION, points=points)
    print(f"{NEW_COLLECTION}: {len(points)} chunks embedded with {NEW_MODEL}")

if __name__ == "__main__":
    build_new_collection()
```

We deliberately reused a small, already-local-friendly model (`all-MiniLM-L12-v2`, ~120MB) rather than forcing a large new download — the migration _mechanism_ you're learning is identical regardless of which model you point it at.

### Step 3 — Prove the danger before you trust the fix

```python
# benchmarks/wrong_model_query.py -- same dimension, different vector space, no error either way
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

client = QdrantClient(url="http://localhost:6333")
query = "why doesn't Qdrant offer an IVF index option?"

v1 = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
v2 = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")

wrong = client.query_points(collection_name="docs_512_v2", query=v1.encode(query).tolist(), limit=3)
correct = client.query_points(collection_name="docs_512_v2", query=v2.encode(query).tolist(), limit=3)

print("WRONG model embedding   -- top score:", round(wrong.points[0].score, 4))
print("CORRECT model embedding -- top score:", round(correct.points[0].score, 4))
```

Both calls succeed — Qdrant checks vector dimensionality (384 == 384), not which model produced it. Compare the scores and the actual top result between the two. The gap you see is real, measured evidence of Concept 3, not a theoretical warning.

### Step 4 — Atomic cutover

```python
# scripts/swap_alias.py
from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333")
client.update_collection_aliases(
    change_aliases_operations=[
        models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name="docs_active")),
        models.CreateAliasOperation(
            create_alias=models.CreateAlias(collection_name="docs_512_v2", alias_name="docs_active")
        ),
    ]
)
print("docs_active now points to docs_512_v2")
```

Both actions execute atomically in one call — no window where the alias points at nothing or at a half-migrated collection. Update `app/embeddings.py` / wherever the query-time embedder is chosen to use `all-MiniLM-L12-v2` going forward, matching whatever `docs_active` now actually points to. Once you've verified search quality against the new collection, `client.delete_collection("docs_512")` retires the old one — keep it around for a rollback window first.

---

## Common Failure Modes

### Failure 1: Search results go quietly bad after a "successful" model migration

**Root Cause:** the alias was swapped (Step 4) but the query-time embedder in `app/embeddings.py` still encodes with the old model. No error anywhere — dimensions still match — just steadily worse retrieval that's easy to misattribute to something else entirely. **Diagnose:** re-run Part G's Step 3 comparison against the newly active collection; if the "wrong model" score is what you're actually getting in production, the embedder swap was missed.

### Failure 2: A shrunk document leaves orphaned chunks behind

**Root Cause:** `update_document.py` was skipped in favor of just calling `add_document.py` again on a shortened file — upsert only touches the IDs you hand it, so old chunk positions that no longer exist in the shorter document are never removed. **Fix:** updates always go through `update_document.py`'s delete-by-filter-then-reinsert, never a bare re-add.

### Failure 3: Delete/update feels slow even on a ~100-point collection

**Root Cause:** Part D's payload index on `source_doc` was skipped, so every filtered delete falls back to a full collection scan. Invisible at this scale, real at the scale you measured in 10.4. **Fix:** run Part D's `create_payload_index` once, before doing any real churn.

### Failure 4: Collection shows `grey` status and stops indexing after a Docker restart

**Root Cause:** restarting the Qdrant container (for example, reverting the `mem_limit` at the end of 10.4) while background optimization was mid-flight leaves the collection's optimizer paused rather than resumed automatically — documented Qdrant behavior, not data loss. **Fix:** send any update operation to kick it back into motion — a no-op `client.update_collection(collection_name="docs_512", optimizers_config=models.OptimizersConfigDiff())` is enough.

---

## Verification Checklist

- [ ] `chunking/stable_ids.py`'s `stable_chunk_id()` depends only on `(source_doc, chunk_index)`, never on document count or order
- [ ] `scripts/add_document.py` on a new file changes `docs_512`'s count by exactly that file's chunk count — nothing else shifts
- [ ] Payload index on `source_doc` exists before any filtered delete/update is run
- [ ] `scripts/update_document.py` deletes old chunks by filter before re-adding — verified by checking chunk count after shrinking a test file
- [ ] `scripts/delete_document.py` removes exactly one document's chunks and nothing else
- [ ] You've watched `indexed_vectors_count` stay near 0 and can explain why that's correct at this scale
- [ ] You've watched `points_count` look momentarily "off" after a churn cycle and can explain why, citing soft-delete + approximate counting
- [ ] The alias-based migration completed: `docs_active` points at `docs_512_v2`, the query-time embedder matches, and Part G Step 3's comparison confirms it

---

## Directory Structure After This Session

```
ai-infra-portfolio/
├── app/
│   ├── rag_pipeline.py             <- queries "docs_active" alias, not a hardcoded name
│   ├── embeddings.py                <- embedder matches whichever model is currently active
│   └── ...
├── chunking/
│   ├── chunker.py                   <- unchanged
│   └── stable_ids.py                <- NEW
├── retrieval/                       <- unchanged from 10.3
├── corpus/                          <- + this session's own notes, added incrementally
├── scripts/
│   ├── load_chunks.py               <- from 10.2, now used only for the very first bulk load
│   ├── add_document.py              <- NEW
│   ├── update_document.py           <- NEW
│   ├── delete_document.py           <- NEW
│   ├── create_alias.py              <- NEW
│   ├── migrate_embedding_model.py   <- NEW
│   └── swap_alias.py                <- NEW
├── benchmarks/
│   ├── ... (10.1-10.4 unchanged)
│   └── wrong_model_query.py         <- NEW
├── docker-compose.yml                 <- back to normal 512m qdrant limit, no special handling needed today
├── docs/
└── requirements.txt
```

---

## Evening Integration — Before You Close the Laptop

```bash
git add chunking/stable_ids.py scripts/add_document.py scripts/update_document.py \
        scripts/delete_document.py scripts/create_alias.py scripts/swap_alias.py \
        scripts/migrate_embedding_model.py benchmarks/wrong_model_query.py \
        app/rag_pipeline.py app/embeddings.py
git commit -m "[pillar-10.5] stable chunk IDs, incremental add/update/delete, alias-based embedding model migration"
git push
```

**3-sentence journal:**

1. What you built today and what it does
2. What broke and the exact error / symptom
3. What you'd do differently if starting over

Pillar 10 is complete. Check it off in your tier tracker.

---

## Commit Message

```
[pillar-10.5] stable chunk IDs, incremental add/update/delete, alias-based embedding model migration
```

---

## Preview: What 11.1 Builds Directly On This

**Pillar 11 (Eval Infrastructure)** starts with **Session 11.1 (Eval Design Fundamentals)** — LLM-as-judge vs. human eval vs. heuristic scoring, and the deceptively hard question "eval for what?" Everything you've measured across Pillar 10 (Recall@5, MRR, NDCG@5, hybrid vs. dense-only, chunk size tradeoffs) was retrieval-quality evaluation. Pillar 11 builds the _general_ eval harness — the one that scores full end-to-end RAG answers, not just whether the right chunk got retrieved — and formalizes the golden query set you've been hand-maintaining since 10.2 into the 50-pair golden dataset the rest of your portfolio's CI gate depends on.

---

## Quick Reference Card

```
STABLE IDS:
  uuid.uuid5(NAMESPACE, f"{source_doc}::{chunk_index}") -> deterministic,
  content-independent, safe for incremental add/update/delete
  NEVER change NAMESPACE once real points exist

SOFT DELETE:
  Qdrant marks deletions with a bitmask, doesn't touch disk immediately
  Vacuum Optimizer cleans up later: deleted_threshold=0.2, vacuum_min_vector_number=1000 (defaults)
  points_count / vectors_count are APPROXIMATE by Qdrant's own documentation
  -- use client.count(name).count when you need an exact number

WHY indexed_vectors_count STAYS ~0 ON A SMALL COLLECTION:
  HNSW segment only builds once unindexed data crosses indexing_threshold
  (20,000 kB default) -- below that, Qdrant deliberately full-scans instead
  Same tradeoff as 10.1's HNSW-vs-exact discussion, from the other side

FILTER-BASED DELETE/UPDATE:
  client.delete(collection_name=, points_selector=models.FilterSelector(
      filter=models.Filter(must=[models.FieldCondition(
          key="source_doc", match=models.MatchValue(value=doc_id))])))
  ALWAYS create a payload index on the filtered field first:
  client.create_payload_index(collection_name=, field_name=, field_schema=models.PayloadSchemaType.KEYWORD)

EMBEDDING MODEL MIGRATION (blue-green + alias):
  1. client.update_collection_aliases(change_aliases_operations=[
       models.CreateAliasOperation(create_alias=models.CreateAlias(
           collection_name=, alias_name="docs_active"))])
  2. Build new collection with new model, same stable ID scheme
  3. Verify: same-dimension vectors from different models are NOT
     interchangeable -- no error, just silently wrong results
  4. Atomic swap: DeleteAliasOperation + CreateAliasOperation in ONE call
  5. Query-time embedder must match whatever docs_active currently points to

GREY COLLECTION STATUS:
  means optimizer paused (usually after a restart mid-optimization), not
  data loss -- send any update_collection() call to resume it
```