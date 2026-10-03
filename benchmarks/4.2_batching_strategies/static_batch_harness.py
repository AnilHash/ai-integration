# static_batch_harness.py
"""
A REAL static-batching implementation. vLLM has no toggle for this -- its
own maintainers confirmed continuous batching "cannot be turned off" and
disabling it "brings no benefit in performance," so it was never built as
an option. This harness exists specifically to give today's comparison
something real to measure against.
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

os.environ["TRANSFORMERS_NO_TF"] = "1"  # or:
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # REQUIRED for decoder-only models -- right-padding
# silently corrupts position embeddings and produces
# garbled output with no error raised

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.float16, device_map="cuda:1"
)
END_TOKEN_ID = tokenizer.convert_tokens_to_ids("<|end|>")
EOS_IDS = [tokenizer.eos_token_id, END_TOKEN_ID]  # 32000 and 32007


def build_static_prompt(question_with_context: str) -> str:
    messages = [{"role": "user", "content": question_with_context}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def run_static_batch(prompts: list[str], max_new_tokens: int = 200) -> dict:
    """
    Pads every prompt in THIS batch to the length of the longest one, runs
    generate() once, and waits for every sequence to finish -- including
    ones that reached an eos token in the first few steps. The padded
    positions still occupy compute and memory for the whole batch's
    lifetime; nothing frees early.
    """
    formatted = [build_static_prompt(p) for p in prompts]
    start = time.perf_counter()
    inputs = tokenizer(formatted, return_tensors="pt", padding=True).to("cuda:1")
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=EOS_IDS,
        )
    elapsed = time.perf_counter() - start

    total_new_tokens = 0
    for output in outputs:
        gen = output[input_len:].tolist()
        real_len = next((j for j, t in enumerate(gen) if t in EOS_IDS), len(gen))
        total_new_tokens += real_len

    return {"elapsed": elapsed, "tokens": total_new_tokens, "batch_size": len(prompts)}


def run_static_batching_sweep(all_prompts: list[str], batch_size: int = 8) -> dict:
    """Sequential fixed-size batches -- batch N+1 does not start until batch N
    is fully done, by definition of static batching."""
    start = time.perf_counter()
    total_tokens = 0
    for i in range(0, len(all_prompts), batch_size):
        batch = all_prompts[i : i + batch_size]
        result = run_static_batch(batch)
        total_tokens += result["tokens"]
        print(
            f"  batch {i // batch_size}: {result['tokens']} tokens in {result['elapsed']:.1f}s"
        )
    wall_time = time.perf_counter() - start
    return {
        "total_tokens": total_tokens,
        "wall_time": wall_time,
        "throughput": total_tokens / wall_time,
    }
