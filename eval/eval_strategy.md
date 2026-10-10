# eval/eval_strategy.md

## Three-Tier Eval Strategy — ai-infra-portfolio

| Tier              | What it catches                              | What it can't see                                  | Cost            |
| ----------------- | -------------------------------------------- | -------------------------------------------------- | --------------- |
| Heuristic         | Structural pattern violations                | Whether the pattern is _correct_, not just present | ~0ms, $0        |
| Retrieval quality | Wrong/missing context reaching the LLM       | What the LLM did with correct context              | ~10ms, $0       |
| LLM-judge         | Hallucination, off-topic, incomplete answers | Its own reliability — must be human-calibrated     | ~1-3s, $0 local |

## Eval-for-what decision table

| The real question                                                       | Right tier                             | Why                                                                                     |
| ----------------------------------------------------------------------- | -------------------------------------- | --------------------------------------------------------------------------------------- |
| "Does it cite a source when it should, and refuse when it can't?"       | **Heuristic**                          | Deterministic, pattern-matchable, no judgment call (2.6)                                |
| "Did retrieval put the right chunk in the context window?"              | **Retrieval quality**                  | Ground-truth chunk IDs exist — a ranking problem with a checkable right answer (10.2)   |
| "Given the right context, is the answer actually correct and complete?" | **LLM-judge**                          | No mechanically-checkable ground truth for open-ended language                          |
| "Did the model invent a claim not present in the context?"              | **LLM-judge** (faithfulness criterion) | Requires reading comprehension across two documents — not a keyword problem             |
| "Is my LLM-judge's opinion worth trusting at all?"                      | **Human eval (you)**                   | Judge validity is itself an empirical claim, not an assumption you get to make for free |

## Judge design decisions and why

- Pointwise, not pairwise -> position bias has no ordering to attach to (proven: position_bias_check.py)
- Reference-anchored, not reference-free -> grading against known-correct answer, not vibes
- Rubric-decomposed (faithful/relevant/complete), not holistic score -> verbosity bias has no length-correlated score to inflate
- Judge model must differ from generator model -> local dev already satisfies this (phi3:mini gen / qwen2.5:1.5b judge);
  CI does NOT yet (qwen2.5:1.5b would grade itself) -- fix in 11.3, use gemma2:2b as CI's judge

## Human calibration status

- [ ] Not yet done. TODO before trusting judge in any CI gate (11.3): hand-label
      10-15 items myself, compare against judge verdicts, record agreement rate here.
