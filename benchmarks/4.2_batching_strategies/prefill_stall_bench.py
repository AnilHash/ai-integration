#!/usr/bin/env python3
"""
Prefill-stall benchmark for vLLM.

Starts N short streaming requests, injects one huge-prompt request mid-stream,
and measures the largest inter-token gap the ongoing streams see around the
injection. Repeats over several trials and compares server configs
(e.g. unchunked vs chunked prefill).

Usage:
  # Launch vLLM for each config automatically, run trials, compare:
  python prefill_stall_bench.py --trials 5

  # Benchmark an already-running server (no launching):
  python prefill_stall_bench.py --base-url http://localhost:8000 --label mine

  # Custom configs (repeatable), "label:extra vllm args":
  python prefill_stall_bench.py \
      --config "chunked-256:--enable-chunked-prefill --max-num-batched-tokens 256" \
      --config "chunked-1024:--enable-chunked-prefill --max-num-batched-tokens 1024"
"""
import argparse
import asyncio
import json
import os
import shlex
import signal
import statistics
import subprocess
import time
import uuid

import httpx

FILLER = (
    "The history of computing spans centuries, from mechanical calculators to modern "
    "accelerators. Engineers balance throughput, latency and memory when serving "
    "language models at scale. "
)

SHORT_QUESTIONS = [
    "Explain how a transformer works.",
    "Write a long story about a lighthouse keeper.",
    "Describe the history of the internet in detail.",
    "Explain how continuous batching works in LLM serving.",
]

# NOTE: flag names vary by vLLM version. In V1 engines chunked prefill is always on
# and "--no-enable-chunked-prefill" may be rejected; if so, drop that flag and keep
# the large --max-num-batched-tokens (>= prompt length), which makes the whole
# prefill run in a single step.
DEFAULT_CONFIGS = {
    "unchunked": ["--no-enable-chunked-prefill", "--max-num-batched-tokens", "4096"],
    "chunked-512": ["--enable-chunked-prefill", "--max-num-batched-tokens", "512"],
}


# ----------------------------------------------------------------------------
# Streaming client
# ----------------------------------------------------------------------------
async def stream_with_timestamps(client, base_url, model, prompt, t0, max_tokens):
    """Return list of timestamps (seconds since shared t0), one per content chunk."""
    times = []
    async with client.stream(
        "POST",
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
            "ignore_eos": True,  # vLLM extension: guarantees streams stay alive
        },
    ) as resp:
        if resp.status_code != 200:
            body = await resp.aread()
            print(f"Request failed: {resp.status_code} - {body[:300]!r}")
            return []
        async for line in resp.aiter_lines():
            if not line.startswith("data: ") or line.strip() == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            choices = chunk.get("choices") or []
            if choices and choices[0].get("delta", {}).get("content"):
                times.append(time.perf_counter() - t0)
    return times


async def count_tokens(client, base_url, model, text):
    """Use vLLM's /tokenize endpoint; fall back to a rough word-based estimate."""
    try:
        r = await client.post(f"{base_url}/tokenize", json={"model": model, "prompt": text})
        r.raise_for_status()
        return r.json()["count"]
    except Exception:
        return int(len(text.split()) * 1.4)


async def build_contexts(client, base_url, args):
    unit_tokens = max(1, await count_tokens(client, base_url, args.model, FILLER))
    short_ctx = FILLER * max(1, args.short_tokens // unit_tokens)
    huge_ctx = FILLER * max(1, args.huge_tokens // unit_tokens)
    huge_total = await count_tokens(client, base_url, args.model, huge_ctx)
    if huge_total + args.injected_tokens + 100 > args.max_model_len:
        raise SystemExit(
            f"Huge prompt (~{huge_total} tok) + {args.injected_tokens} output exceeds "
            f"max_model_len={args.max_model_len}. Lower --huge-tokens or --injected-tokens."
        )
    print(f"  prompt sizes: short ~{args.short_tokens} tok, huge ~{huge_total} tok")
    return short_ctx, huge_ctx


# ----------------------------------------------------------------------------
# One trial
# ----------------------------------------------------------------------------
async def run_trial(client, base_url, args, short_ctx, huge_ctx):
    t0 = time.perf_counter()
    ongoing = [
        asyncio.create_task(
            stream_with_timestamps(
                client, base_url, args.model,
                f"{short_ctx}\n\nQuestion: {q}", t0, args.ongoing_tokens,
            )
        )
        for q in SHORT_QUESTIONS * 2
    ]
    await asyncio.sleep(args.inject_at)

    t_inject = time.perf_counter() - t0
    # Unique prefix defeats prefix caching so the prefill is real work.
    huge_prompt = (
        f"[{uuid.uuid4()}] {huge_ctx}\n\nQuestion: Summarize everything above in detail."
    )
    injected = asyncio.create_task(
        stream_with_timestamps(
            client, base_url, args.model, huge_prompt, t0, args.injected_tokens
        )
    )
    results = await asyncio.gather(*ongoing, injected)
    return t_inject, results[:-1], results[-1]


def analyze(t_inject, ongoing, injected, window):
    """Per-trial metrics. Only streams still alive at injection are counted."""
    baseline, stalls = [], []
    for times in ongoing:
        if len(times) < 2 or times[-1] <= t_inject:
            continue  # finished before the injection; says nothing about the stall
        gaps = [(a, b, b - a) for a, b in zip(times, times[1:])]
        baseline += [g for a, b, g in gaps if b < t_inject]
        in_window = [g for a, b, g in gaps if b >= t_inject and a <= t_inject + window]
        if in_window:
            stalls.append(max(in_window))
    ttft = (injected[0] - t_inject) if injected else float("nan")
    return {"baseline": baseline, "stalls": stalls, "ttft": ttft, "t_inject": t_inject}


# ----------------------------------------------------------------------------
# Stats / reporting
# ----------------------------------------------------------------------------
def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def summarize(label, trials):
    base = [g for t in trials for g in t["baseline"]]
    stalls = [s for t in trials for s in t["stalls"]]
    ttfts = [t["ttft"] for t in trials if t["ttft"] == t["ttft"]]
    s = {
        "label": label,
        "trials": len(trials),
        "live_samples": len(stalls),
        "baseline_ms": statistics.median(base) * 1000 if base else float("nan"),
        "stall_median_ms": statistics.median(stalls) * 1000 if stalls else float("nan"),
        "stall_p95_ms": pct(stalls, 95) * 1000,
        "stall_max_ms": max(stalls) * 1000 if stalls else float("nan"),
        "ttft_median_ms": statistics.median(ttfts) * 1000 if ttfts else float("nan"),
        "ttft_min_ms": min(ttfts) * 1000 if ttfts else float("nan"),
        "ttft_max_ms": max(ttfts) * 1000 if ttfts else float("nan"),
    }
    s["ratio"] = s["stall_median_ms"] / s["baseline_ms"] if s["baseline_ms"] else float("nan")
    return s


def print_summary(rows):
    print("\n" + "=" * 100)
    print(f"{'config':<14}{'trials':>7}{'samples':>9}{'base ms':>9}"
          f"{'stall med':>11}{'stall p95':>11}{'stall max':>11}{'x base':>8}"
          f"{'inj TTFT med [min-max] ms':>30}")
    print("-" * 100)
    for r in rows:
        print(f"{r['label']:<14}{r['trials']:>7}{r['live_samples']:>9}{r['baseline_ms']:>9.0f}"
              f"{r['stall_median_ms']:>11.0f}{r['stall_p95_ms']:>11.0f}{r['stall_max_ms']:>11.0f}"
              f"{r['ratio']:>8.1f}"
              f"{r['ttft_median_ms']:>14.0f} [{r['ttft_min_ms']:.0f}-{r['ttft_max_ms']:.0f}]")
    print("=" * 100)
    print("samples = (live stream, trial) pairs that overlapped the injection.")
    print("stall = max inter-token gap in [inject, inject+window]; base = median gap before injection.")


# ----------------------------------------------------------------------------
# Server management
# ----------------------------------------------------------------------------
def launch_server(model, port, max_model_len, extra_args, log_path):
    cmd = [
        "vllm", "serve", model,
        "--port", str(port),
        "--max-model-len", str(max_model_len),
        "--no-enable-prefix-caching",
        *extra_args,
    ]
    print("  launching:", " ".join(cmd))
    log = open(log_path, "w")
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


async def wait_ready(base_url, proc, log_path, timeout=900):
    t_start = time.time()
    async with httpx.AsyncClient() as c:
        while time.time() - t_start < timeout:
            if proc.poll() is not None:
                raise RuntimeError(f"vLLM exited early (code {proc.returncode}); see {log_path}")
            try:
                if (await c.get(f"{base_url}/health")).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(2)
    raise TimeoutError(f"server not ready after {timeout}s; see {log_path}")


def stop_server(proc):
    if proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    time.sleep(5)  # let the GPU free its memory before the next launch


# ----------------------------------------------------------------------------
# Per-config driver
# ----------------------------------------------------------------------------
async def benchmark_config(label, base_url, args):
    print(f"\n### {label}")
    trials = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
        short_ctx, huge_ctx = await build_contexts(client, base_url, args)
        for i in range(args.warmup + args.trials):
            t_inject, ongoing, injected = await run_trial(client, base_url, args, short_ctx, huge_ctx)
            m = analyze(t_inject, ongoing, injected, args.inject_window)
            tag = "warmup" if i < args.warmup else f"trial {i - args.warmup + 1}"
            worst = max(m["stalls"]) * 1000 if m["stalls"] else float("nan")
            print(f"  [{tag}] live streams={len(m['stalls'])}/{len(ongoing)}  "
                  f"worst stall={worst:.0f}ms  injected TTFT={m['ttft'] * 1000:.0f}ms")
            if i >= args.warmup:
                trials.append(m)
            await asyncio.sleep(args.cooldown)
    return trials


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="microsoft/Phi-3-mini-4k-instruct")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--base-url", help="benchmark an already-running server instead of launching")
    ap.add_argument("--label", default="external", help="label to use with --base-url")
    ap.add_argument("--config", action="append",
                    help='"label:extra vllm args" (repeatable); default = unchunked vs chunked-512')
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1, help="discarded trials per config")
    ap.add_argument("--inject-at", type=float, default=1.0)
    ap.add_argument("--inject-window", type=float, default=3.0, help="seconds after injection to inspect")
    ap.add_argument("--ongoing-tokens", type=int, default=400)
    ap.add_argument("--injected-tokens", type=int, default=200)
    ap.add_argument("--short-tokens", type=int, default=300, help="approx prompt size of ongoing requests")
    ap.add_argument("--huge-tokens", type=int, default=3000, help="approx prompt size of injected request")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--cooldown", type=float, default=2.0)
    ap.add_argument("--out", default="prefill_stall_results.json")
    args = ap.parse_args()

    summaries, raw = [], {}

    if args.base_url:
        trials = await benchmark_config(args.label, args.base_url.rstrip("/"), args)
        summaries.append(summarize(args.label, trials))
        raw[args.label] = trials
    else:
        configs = dict(DEFAULT_CONFIGS)
        if args.config:
            configs = {}
            for spec in args.config:
                name, _, extra = spec.partition(":")
                configs[name] = shlex.split(extra)
        base_url = f"http://localhost:{args.port}"
        for label, extra in configs.items():
            log_path = f"vllm_{label}.log"
            proc = launch_server(args.model, args.port, args.max_model_len, extra, log_path)
            try:
                await wait_ready(base_url, proc, log_path)
                trials = await benchmark_config(label, base_url, args)
                summaries.append(summarize(label, trials))
                raw[label] = trials
            finally:
                stop_server(proc)

    print_summary(summaries)
    with open(args.out, "w") as f:
        json.dump({"summaries": summaries, "raw": raw}, f, indent=2)
    print(f"\nraw results written to {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
