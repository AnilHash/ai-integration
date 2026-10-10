"""
Empirical proof that naive pairwise LLM-judging is order-sensitive on this model. Run this BEFORE trusting any pairwise judge design.
"""

from dotenv import load_dotenv

load_dotenv()

from langfuse.openai import openai  # type: ignore
from app.rag_pipeline import run_rag_pipeline

OLLAMA_BASE_URL = "http://localhost:11434/v1"
JUDGE_MODEL = "qwen2.5:1.5b"

llm_client = openai.OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

PAIRWISE_PROMPT = """
You are comparing two answers to the same question.
Question: {query}
Answer A: {answer_a}
Answer B: {answer_b}
Which answer is better? Respond with exactly one word: "A" or "B".
"""


def judge_pairwise(query: str, answer_a: str, answer_b: str) -> str:
    resp = llm_client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {
                "role": "user",
                "content": PAIRWISE_PROMPT.format(
                    query=query, answer_a=answer_a, answer_b=answer_b
                ),
            }
        ],
        temperature=0,
    )
    return (
        resp.choices[0].message.content.strip()
        if resp.choices[0].message.content is not None
        else ""
    )


if __name__ == "__main__":
    query = "What is a pydantic?"

    result_v1 = run_rag_pipeline(query, prompt_version=1)
    result_v2 = run_rag_pipeline(query, prompt_version=2)
    print(result_v1["answer"])
    print(result_v2["answer"])
    forward = judge_pairwise(query, result_v1["answer"], result_v2["answer"])
    swapped = judge_pairwise(query, result_v2["answer"], result_v1["answer"])

    print(f"v1=A, v2=B -> judge picked {forward}")
    print(f"v2=A, v1=B -> judge picked {swapped}")
    print()

    print("Unbiased judge: the SAME underlying answer wins both times, under")
    print("whichever letter it happened to get. If 'A' wins both runs (or 'B'")
    print("wins both runs), the judge is tracking the LETTER, not the CONTENT.")
