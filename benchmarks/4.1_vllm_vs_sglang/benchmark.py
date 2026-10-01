# %%writefile benchmark.py
# benchmark.py — reused against both engines, same OpenAI-compatible wire format
import asyncio
import time
import json
import httpx

SHARED_CONTEXT = (
    """You are a helpful assistant answering questions about a software
documentation corpus. Use only the following context to answer.

Context:
[doc-001] FastAPI is a modern, high-performance web framework for building
APIs with Python, based on standard type hints. It uses Pydantic models for
request and response validation, and provides automatic interactive API
documentation.
[doc-002] Pydantic is a data validation library for Python that uses type
annotations to validate data at runtime, raising structured errors when
input doesn't match the expected shape.
[doc-003] Docker packages an application together with its dependencies and
runtime into a single, portable, isolated container that behaves
consistently regardless of the host environment.
"""
    * 3
)  # repeated to approximate a realistic ~500-800 token retrieved-context block

QUESTIONS = [
    "What does FastAPI use for request validation?",
    "How does Pydantic validate data?",
    "What problem does Docker solve?",
    "Explain FastAPI's approach to API documentation.",
    "What happens when Pydantic input doesn't match the expected shape?",
    # ... extend to 20 for a real run; these five are enough to prove the script works
]


def build_prompt(question: str) -> str:
    return f"{SHARED_CONTEXT}\n\nQuestion: {question}"


async def single_request(
    client: httpx.AsyncClient, base_url: str, question: str
) -> dict:
    start = time.perf_counter()
    resp = await client.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": "microsoft/Phi-3-mini-4k-instruct",
            # "messages": [{"role": "user", "content": build_prompt(question)}],
            "messages": [{"role": "user", "content": question}],
            "max_tokens": 200,
            "temperature": 0,
        },
        timeout=120,
    )
    elapsed = time.perf_counter() - start
    data = resp.json()
    return {"elapsed": elapsed, "tokens": data["usage"]["completion_tokens"]}


async def run_concurrency_level(
    base_url: str, concurrency: int, prompts, total_requests: int = 20
) -> dict:
    async with httpx.AsyncClient() as client:
        sem = asyncio.Semaphore(concurrency)

        async def bound(i: int):
            q = prompts[i % len(prompts)]
            async with sem:
                return await single_request(client, base_url, q)

        start = time.perf_counter()
        results = await asyncio.gather(*[bound(i) for i in range(total_requests)])
        wall_time = time.perf_counter() - start

    total_tokens = sum(r["tokens"] for r in results)
    latencies = sorted(r["elapsed"] for r in results)
    return {
        "concurrency": concurrency,
        "throughput_tokens_per_sec": total_tokens / wall_time,
        "p50_latency_s": latencies[len(latencies) // 2],
        "p95_latency_s": latencies[int(len(latencies) * 0.95)],
    }


async def main(base_url: str, engine_name: str, output_path: str, prompts: list[str]):
    results = []
    for concurrency in [1, 4, 8, 16]:
        r = await run_concurrency_level(
            base_url, concurrency, prompts=prompts, total_requests=len(prompts)
        )
        r["engine"] = engine_name
        print(r)
        results.append(r)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    return results
