import os

from pydantic import BaseModel, Field

from langfuse.openai import openai  # type: ignore

OLLAMA_BASE_URL = "http://localhost:11434/v1"
JUDGE_MODEL = os.getenv("OLLAMA_JUDGE_MODEL", "qwen2.5:1.5b")
llm_client = openai.OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")


class JudgeVerdict(BaseModel):
    faithful: bool = Field(
        description="Every claim in the candidate answer is supported by the retrieved context - no invented facts"
    )
    relevant: bool = Field(
        description="The candidate answer actually addresses the question asked"
    )
    complete: bool = Field(
        description="The candidate answer covers what the reference answer covers, not a subset of it"
    )
    reasoning: str = Field(
        description="One sentence justifying the three verdicts above"
    )


JUDGE_PROMPT = """ You are grading a candidate answer against a reference answer and the source context it was generated from. Do not reward style, tone, or length - grade only the three criteria in the schema.

Question: {query}

Retrieved context:
{context}

Reference answer (known-correct):
{reference}

Candidate answer (to grade): 
{candidate}

Evaluate the candidate strictly against the criteria in the schema."""


def judge_answer(
    query: str, context: str, reference: str, candidate: str
) -> JudgeVerdict:
    resp = llm_client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {
                "role": "user",
                "content": JUDGE_PROMPT.format(
                    query=query,
                    context=context,
                    reference=reference,
                    candidate=candidate,
                ),
            }
        ],
        temperature=0,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "judge_verdict",
                "schema": JudgeVerdict.model_json_schema(),
            },
        },
    )
    raw = resp.choices[0].message.content
    if raw is None:
        return 0
    try:
        return JudgeVerdict.model_validate_json(raw)
    except Exception as e:
        raise ValueError(f"Judge returned invalid structured output: {raw!r}") from e


def judge_answer_safe(
    query: str, context: str, reference: str, candidate: str, max_attempts: int = 2
) -> JudgeVerdict:
    """
    Schema constraint reduces malformed output, it doesn't eliminate it. One retry is enough for a CPU-quantized model at temp=0.
    """
    last_error = None
    for _ in range(max_attempts):
        try:
            return judge_answer(query, context, reference, candidate)
        except ValueError as e:
            last_error = e
    raise last_error
