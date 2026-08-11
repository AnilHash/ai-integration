

**Pillar 2 — Observability | Tier 1 | Run Locally**

> **Where you are:** 2.5 gave you drift detection — a way to notice quality degrading _after_ it's already in production, by watching a trend. Today you build the earlier line of defense: a fixed, curated test set that runs on every pull request and blocks a regression from ever merging in the first place. Different data source (curated, not live traffic), different trigger (a PR event, not a time window), same underlying idea — quality is a thing you measure, not a thing you assume.

---

## Session Goals

Right now, nothing stops a bad prompt edit or a broken pipeline change from merging straight to `main`. 2.4 gave you a scorer. 2.5 gave you a way to watch it drift over live traffic. Today those pieces become a gate: a small, deliberately-designed set of test questions — including ones your retrieval _can't_ answer, testing whether the system honestly says so instead of hallucinating — runs automatically in GitHub Actions on every PR, and fails the build if quality drops below a floor.

**A scope note up front, worth reading before you start:** the curriculum's literal lab description mentions "Recall@5" — a retrieval-quality metric. Your pipeline still uses 2.1's mock retrieval (three hardcoded chunks, identical for every query), so a real Recall@5 would be either meaningless or trivially 100%. Building a fake metric to check a box would teach the wrong lesson. Instead, today builds the **CI gate infrastructure** — which is the actual point of this session — using metrics that are honestly meaningful _right now_. Pillar 10.2 swaps in real Recall@5 against real retrieval later; the harness structure you build today doesn't change, only the specific metric does.

**The deliverable:** A 12-item golden set (`eval/golden_set.json`), a runner that gates on an absolute pass-rate floor (`eval/run_eval.py`), and a GitHub Actions workflow that runs it on every PR against `main` — with Ollama installed and a small model pulled fresh inside the CI runner itself.

---

## The 3 Concepts You Must Own After This Session

1. **What an eval harness actually is** — the five components (dataset, runner, scorers, aggregator, gate) and how they compose, independent of which specific metric you're computing.
2. **What makes a good golden set** — representative cases _and_ deliberate edge cases, small enough to run fast in CI, reviewable like code, and sized honestly against what statistics can support (a 12-item set can't support a significance test the way 2.4's 50-per-variant experiment could — you need an absolute floor instead, and knowing why is the point).
3. **Running your own model inside CI** — there's no Ollama instance sitting in GitHub's cloud waiting for you. Today's CI job installs Ollama and pulls a model fresh inside the runner, and deliberately uses a _different_, smaller model than local dev — a real speed/fidelity tradeoff you should be able to name, not something to hide.

---

## What Is an Eval Harness?

Every eval harness — however simple or elaborate — has the same five parts. Knowing this decomposition is what lets you scale from today's 12-item CI gate to Pillar 11's fuller harness without re-learning the shape of the problem:

```
┌──────────────┐   ┌─────────┐   ┌──────────┐   ┌─────────────┐   ┌──────┐
│   DATASET     │──►│ RUNNER  │──►│ SCORERS  │──►│ AGGREGATOR  │──►│ GATE │
│              │   │         │   │          │   │             │   │      │
│ golden_set   │   │ calls   │   │ cites_   │   │ pass rate   │   │ floor│
│ .json        │   │ run_rag_│   │ source() │   │ across all  │   │ check│
│ 12 items     │   │ pipeline│   │ is_       │   │ items       │   │ →    │
│              │   │ per item│   │ grounded_│   │             │   │ exit │
│              │   │         │   │ refusal()│   │             │   │ code │
└──────────────┘   └─────────┘   └──────────┘   └─────────────┘   └──────┘
     "what to           "run             "was it         "how did      "should
      test"           each test"        correct?"       we do overall?" this ship?"
```

Every eval harness you'll ever build — heuristic, LLM-as-judge, retrieval quality, whatever — is this same five-stage pipeline with different implementations slotted into each stage. Today's is deliberately the simplest possible version of each stage.

---

## What Makes a Good Golden Set

A golden set that only contains questions your system was obviously built to answer proves nothing — it'll pass even after a real regression, because it never exercises the failure mode that regression introduced. A good golden set is built from two deliberately different categories:

1. **Representative cases** — the kind of question you actually expect, testing the happy path stays happy.
2. **Deliberate edge cases** — questions chosen specifically to probe a known failure mode, even if they're rare in real traffic. Today's edge case: questions the retrieved context _cannot_ answer, testing whether the model honestly refuses instead of confidently hallucinating using outside knowledge. This directly previews Pillar 11.4's "hallucination guard... refusal on OOS [out-of-scope]" — you're building a lightweight version of exactly that concept now, because it's genuinely testable with the pipeline you already have, unlike Recall@5.

A few other properties worth naming explicitly:

- **Checked into git, reviewed like code.** Adding or changing a golden item should go through a PR, same as any other change — it's a test fixture, treat it with the same rigor.
- **Small enough to run fast in CI.** At 10-30 seconds per item on CPU inference, a golden set needs to stay in the 10-20 item range to keep CI runs reasonable — this genuinely constrains golden set size in a way a cloud-hosted, fast-inference eval wouldn't be constrained.
- **Sized honestly against what statistics can support.** 2.4 ran 50 requests per variant specifically so a two-proportion z-test would mean something. A 12-item golden set is too small for that same rigor — a single flaky item swings the pass rate by 8 percentage points. Today's gate uses a simple **absolute pass-rate floor** instead of a statistical comparison against a baseline, precisely because a significance test at n=12 would be unreliable, not because significance testing stopped mattering. Pillar 11.2's 50-item golden set is large enough to support the statistical approach properly.

---

## Part A: Build the Golden Set

Create `eval/golden_set.json`:

```json
[
  {"id": "in-01", "query": "What is FastAPI?", "expect_in_scope": true},
  {"id": "in-02", "query": "What does Pydantic do?", "expect_in_scope": true},
  {"id": "in-03", "query": "How do Docker containers work?", "expect_in_scope": true},
  {"id": "in-04", "query": "Explain what FastAPI is used for", "expect_in_scope": true},
  {"id": "in-05", "query": "What is data validation in Pydantic?", "expect_in_scope": true},
  {"id": "in-06", "query": "What do Docker containers package together?", "expect_in_scope": true},
  {"id": "oos-01", "query": "What is the capital of France?", "expect_in_scope": false},
  {"id": "oos-02", "query": "How do I file my taxes this year?", "expect_in_scope": false},
  {"id": "oos-03", "query": "What's the weather like today?", "expect_in_scope": false},
  {"id": "oos-04", "query": "Who won the last World Cup?", "expect_in_scope": false},
  {"id": "oos-05", "query": "What's a good recipe for pasta carbonara?", "expect_in_scope": false},
  {"id": "oos-06", "query": "How old is the universe?", "expect_in_scope": false}
]
```

Six questions the mock context (FastAPI/Pydantic/Docker chunks) can genuinely answer. Six it can't. **Expected behavior differs by category** — that asymmetry is the entire test.

---

## Part B: A Second Heuristic — Grounded Refusal

2.4 built `cites_source()`. Today adds its counterpart: does the model correctly _decline_ to answer when the context doesn't support it, instead of reaching for outside knowledge it shouldn't be using?

```python
import re

CITATION_PATTERN = re.compile(r"doc-\d{3}")

REFUSAL_PATTERNS = [
    r"don'?t (?:have|contain) enough information",
    r"does not contain enough information",
    r"doesn'?t contain enough information",
    r"cannot answer",
    r"can'?t answer",
    r"context (?:doesn'?t|does not) (?:provide|contain)",
    r"not (?:enough|sufficient) information",
    r"unable to (?:answer|determine)",
]


def cites_source(answer: str) -> bool:
    return bool(CITATION_PATTERN.search(answer))


def is_grounded_refusal(answer: str) -> bool:
    lowered = answer.lower()
    return any(re.search(p, lowered) for p in REFUSAL_PATTERNS)
```

**Be honest about this heuristic's limits.** Keyword matching will miss a refusal phrased in a way you didn't anticipate, and phi3:mini's phrasing isn't perfectly consistent even at low temperature. A semantic check (does this response _mean_ "I don't know," regardless of exact wording) would be more robust — that's exactly the "heuristic vs LLM-as-judge" tradeoff Pillar 11.1 covers properly. Today's version is intentionally the cheap, fast, slightly-imprecise tier of a system that Pillar 11 will make more rigorous.

---

## Part C: Make the Model Configurable

CI is going to use a different, smaller model than your local dev setup — which means the model name can't stay hardcoded. Update `app/llm_client.py`:

```python
import os

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "phi3:mini")
```

Local dev keeps working exactly as before (no `OLLAMA_MODEL` env var set → falls back to `phi3:mini`). CI will set `OLLAMA_MODEL=qwen2.5:1.5b` explicitly — see Part E for why.

---

## Part D: The Eval Runner

Create `eval/run_eval.py`:

```python
"""
CI regression gate. Runs the golden set against whatever prompt is
currently labeled 'production', scores each item, and fails (non-zero
exit code) if the pass rate drops below an absolute floor.

Calls run_rag_pipeline() directly, in-process — NOT through the /query
HTTP endpoint. Two consequences worth knowing:
  1. No running FastAPI server needed in CI — just the Python process.
  2. This never touches Postgres/request_log (only /query does that),
     so CI eval runs don't pollute the production drift-detection data
     from Session 2.5. No database needs to run in CI at all.
"""
import json
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from app.rag_pipeline import run_rag_pipeline
from app.instrumentation import langfuse

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.json"
PASS_RATE_FLOOR = 0.83  # 10 of 12 — tune this as your golden set grows

CITATION_PATTERN = re.compile(r"doc-\d{3}")
REFUSAL_PATTERNS = [
    r"don'?t (?:have|contain) enough information",
    r"does not contain enough information",
    r"doesn'?t contain enough information",
    r"cannot answer",
    r"can'?t answer",
    r"context (?:doesn'?t|does not) (?:provide|contain)",
    r"not (?:enough|sufficient) information",
    r"unable to (?:answer|determine)",
]


def cites_source(answer: str) -> bool:
    return bool(CITATION_PATTERN.search(answer))


def is_grounded_refusal(answer: str) -> bool:
    lowered = answer.lower()
    return any(re.search(p, lowered) for p in REFUSAL_PATTERNS)


def evaluate_item(item: dict) -> dict:
    result = run_rag_pipeline(query=item["query"], user_id="ci-eval", prompt_version=None)
    answer = result["answer"]

    if item["expect_in_scope"]:
        passed = cites_source(answer) and not is_grounded_refusal(answer)
    else:
        passed = is_grounded_refusal(answer)

    return {"id": item["id"], "query": item["query"], "answer": answer, "passed": passed}


def main():
    golden_set = json.loads(GOLDEN_SET_PATH.read_text())
    results = [evaluate_item(item) for item in golden_set]

    passed_count = sum(r["passed"] for r in results)
    total = len(results)
    pass_rate = passed_count / total

    print(f"\n{'=' * 50}\nGOLDEN SET RESULTS: {passed_count}/{total} ({pass_rate:.1%})\n{'=' * 50}")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}] {r['id']}: {r['query']}")
        if not r["passed"]:
            print(f"         answer: {r['answer'][:150]}")

    langfuse.flush()

    if pass_rate < PASS_RATE_FLOOR:
        print(f"\n✗ Pass rate {pass_rate:.1%} is below the floor of {PASS_RATE_FLOOR:.1%} — failing CI")
        sys.exit(1)

    print(f"\n✓ Pass rate {pass_rate:.1%} meets the floor of {PASS_RATE_FLOOR:.1%}")
    sys.exit(0)


if __name__ == "__main__":
    main()
```

### Run It Locally First

Don't debug this for the first time inside CI — verify it works on your machine before wiring up the workflow:

```bash
python eval/run_eval.py
```

You should see all 12 items print PASS/FAIL, a summary line, and exit code `0`. Check with `echo $?` (or `echo $LASTEXITCODE` in PowerShell) after it finishes.

---

## Part E: Running Ollama Inside GitHub Actions

GitHub's CI runners are clean, ephemeral Ubuntu VMs — there is no Ollama instance waiting for you there. Every workflow run needs to install Ollama and pull a model from scratch (or from cache), inside the job itself, before the eval script can call it.

### The Model Tradeoff — Named Explicitly, Not Hidden

||phi3:mini (local dev default)|qwen2.5:1.5b (today's CI choice)|
|---|---|---|
|Size|2.2 GB|1.0 GB|
|Fidelity to what ships|Exact match|A smaller, faster proxy — not identical behavior|
|CI download + inference time|Slower|Faster|
|Why it matters|This is what real users actually hit|This is what CI actually tests|

This is a genuine, real tradeoff in production MLOps — using a cheaper proxy model in CI to keep builds fast, accepting that CI's pass/fail doesn't perfectly mirror production's exact behavior. For a regression gate checking _structural_ behavior (does it cite sources when it should, does it refuse when it should) rather than fine-grained answer quality, a smaller model is a reasonable proxy. If you ever suspect CI and production are diverging meaningfully, the fix is running the same golden set against phi3:mini locally as a periodic cross-check — not something to automate today, just know it's available.

### The Workflow

Create `.github/workflows/eval.yml`:

```yaml
name: Eval Gate

on:
  pull_request:
    branches: [main]

jobs:
  golden-set-eval:
    runs-on: ubuntu-latest
    timeout-minutes: 15
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install Python dependencies
        run: pip install -r requirements.txt

      - name: Cache Ollama models
        uses: actions/cache@v4
        with:
          path: ~/.ollama/models
          key: ollama-qwen2.5-1.5b

      - name: Install Ollama
        run: curl -fsSL https://ollama.com/install.sh | sh

      - name: Start Ollama and pull model
        run: |
          ollama serve &
          for i in $(seq 1 30); do
            curl -s http://localhost:11434/api/tags > /dev/null && break
            sleep 1
          done
          ollama pull qwen2.5:1.5b

      - name: Run golden set eval
        env:
          OLLAMA_MODEL: qwen2.5:1.5b
          LANGFUSE_PUBLIC_KEY: ${{ secrets.LANGFUSE_PUBLIC_KEY }}
          LANGFUSE_SECRET_KEY: ${{ secrets.LANGFUSE_SECRET_KEY }}
          LANGFUSE_HOST: ${{ secrets.LANGFUSE_HOST }}
        run: python eval/run_eval.py
```

**The retry loop before `ollama pull`** matters: `ollama serve &` starts the server in the background, but it needs a moment to be ready to accept connections. Pulling or calling the model before it's up would fail with a connection-refused error. The loop polls `/api/tags` every second, up to 30 seconds, before proceeding.

**The cache step** avoids re-downloading a ~1GB model on every single CI run — subsequent runs restore it from GitHub's cache instead, meaningfully cutting run time after the first execution.

### Configure Secrets and Branch Protection — Two Manual Steps

1. **Repo secrets:** Settings → Secrets and variables → Actions → New repository secret. Add `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`. If you skip this, the eval still runs correctly — per 2.1's resilient design, a missing/invalid Langfuse config disables tracing gracefully rather than crashing the app — you just won't see these CI runs' traces in Langfuse.
    
2. **Branch protection (easy to miss, and without it this gate does nothing):** Settings → Branches → Add branch protection rule for `main` → enable "Require status checks to pass before merging" → select `golden-set-eval`. A workflow that runs and fails doesn't block anything on its own — GitHub only enforces it as a merge requirement if you explicitly tell it to.
    

---

## Common Failure Modes

### Failure 1: `ollama pull` fails with connection refused in CI

**Root Cause:** the eval step (or the pull step) ran before `ollama serve` was actually ready to accept connections — the readiness loop either wasn't included or didn't wait long enough. **Fix:** confirm the retry loop in the workflow is present and check the Actions log to see how many iterations it took; if it's regularly needing close to 30, extend the timeout.

### Failure 2: CI eval fails but the same golden set passes locally

**Two likely causes, check both:**

1. **Model mismatch.** Confirm `OLLAMA_MODEL=qwen2.5:1.5b` is actually set in the workflow's env block, and that `llm_client.py`'s `os.getenv("OLLAMA_MODEL", "phi3:mini")` fallback is in place. If this env var isn't wired through, CI silently tries to call `phi3:mini`, which was never pulled in the CI job, and every request fails outright.
2. **Genuine model behavior difference.** qwen2.5:1.5b is smaller than phi3:mini and won't follow instructions as reliably — a lower pass rate in CI than local isn't necessarily a bug, it may be the real fidelity tradeoff from Part E showing up. If CI consistently runs a few points below local, that's worth knowing, not necessarily worth "fixing" by forcing phi3:mini into CI and accepting the slower runs.

### Failure 3: A single golden item flips between pass and fail across runs, nothing else changed

**Root Cause:** LLM output is stochastic, even at `temperature=0.1` — a borderline item can land on either side of a heuristic check run to run. At n=12, one flaky item is worth ~8 percentage points of pass rate, which can push you across the floor in either direction on pure noise. **This is a known, real problem, not something to hand-debug per flaky item today.** Pillar 11.3 ("CI/CD for AI — Eval Gates... handling flaky evals") builds proper handling for this — retry-and-majority-vote patterns, distinguishing persistent regressions from one-off noise. For now: if a failure is isolated to one item and doesn't reproduce on a re-run, treat it as noise, not signal, and move on.

### Failure 4: Branch protection is configured, but a clearly-failing PR still merges

**Root Cause:** almost always that the status check name selected in branch protection settings doesn't exactly match the job name in the workflow (`golden-set-eval`) — a typo, or the workflow was renamed after protection was configured. **Fix:** re-check Settings → Branches, and confirm the exact job name (not the workflow file name) appears in the required checks list.

---

## Verification Checklist

- [x] `eval/golden_set.json` has 6 in-scope and 6 out-of-scope items
- [x] `is_grounded_refusal()` correctly flags a manually-tested out-of-scope answer
- [x] `llm_client.py`'s `DEFAULT_MODEL` reads from `OLLAMA_MODEL` env var with a `phi3:mini` fallback
- [x] `python eval/run_eval.py` runs successfully on your local machine first
- [x] `.github/workflows/eval.yml` created; readiness retry loop present before `ollama pull`
- [x] Langfuse secrets added to repo settings (or explicitly skipped, knowingly)
- [x] Opened a real PR and watched the Actions tab run the job end-to-end
- [x] Branch protection configured to require `golden-set-eval` before merge
- [x] Deliberately broke something (e.g., temporarily reverted production to a prompt with no citation instruction) and confirmed the PR gate actually fails
- [x] You can explain why today's gate uses an absolute floor instead of a significance test against a baseline

---

## Directory Structure After This Session

```
ai-infra-portfolio/
├── app/
│   ├── main.py
│   ├── db.py
│   ├── instrumentation.py
│   ├── llm_client.py         ← DEFAULT_MODEL now reads OLLAMA_MODEL env var
│   └── rag_pipeline.py
├── eval/                       ← NEW
│   ├── golden_set.json         ← 12 curated test cases
│   └── run_eval.py             ← CI regression gate
├── .github/
│   └── workflows/
│       └── eval.yml            ← NEW — runs eval/run_eval.py on every PR
├── scripts/
├── .env
└── requirements.txt             ← unchanged — no new Python packages
```

---

## Evening Integration — Before You Close the Laptop

```bash
git add eval/ .github/workflows/eval.yml app/llm_client.py
git commit -m "[pillar-2.6] add golden test set + CI eval gate via GitHub Actions"
git push
```

**3-sentence journal:**

1. What you built today and what it does
2. What broke and the exact error / symptom
3. What you'd do differently if starting over

---

## Commit Message

```
[pillar-2.6] add golden test set + CI eval gate via GitHub Actions
```

---

## Preview: What 2.7 Builds Directly On This

**Session 2.7 (Alerting Design for AI Systems)** is where every prior session's individual signal — 2.2's cost, 2.3's latency, 2.5's drift, and today's CI gate — gets assembled into one coherent alert matrix, with a real answer to "what deserves a 3am page versus what can wait for tomorrow morning." Today's gate is a _pre-merge_ check — it stops a bad change from shipping. 2.7 is about _post-deploy_ monitoring — catching the things a 12-item golden set structurally can't, because they only show up under real traffic patterns, at real scale, after the code already merged.

---

## Quick Reference Card

```
EVAL HARNESS — FIVE STAGES:
  Dataset → Runner → Scorers → Aggregator → Gate
  (golden_set.json) (run_rag_pipeline) (cites_source, is_grounded_refusal) (pass rate) (floor check → exit code)

GOLDEN SET DESIGN:
  Representative cases + deliberate edge cases (today: out-of-scope refusal tests)
  Checked into git, reviewed like code
  Small enough for CI speed (10-20 items at CPU inference speed)
  Too small for a significance test → use an absolute floor instead (today)
  Pillar 11.2's 50-item set is large enough for statistical comparison

CI-SPECIFIC GOTCHAS:
  No Ollama in GitHub's cloud — install + pull fresh (or cached) every run
  ollama serve needs a readiness retry loop before ollama pull / any call
  Model name must be env-var-driven (OLLAMA_MODEL), not hardcoded —
    CI and local dev legitimately use different models
  Calling run_rag_pipeline() directly (not /query) → no DB needed in CI,
    no request_log pollution from synthetic CI traffic
  Branch protection must explicitly require the check — a failing
    workflow doesn't block merges on its own

MODEL TRADEOFF IN CI:
  Smaller/faster model (qwen2.5:1.5b) = fast CI, imperfect fidelity to prod
  Same model as prod (phi3:mini) = slower CI, exact fidelity
  Name this tradeoff explicitly — don't pretend CI and prod behave identically
```