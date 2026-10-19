import numpy as np
from langfuse import get_client
from app.db import get_pool
from app.embeddings import get_embedder

RETRY_WINDOW_SECONDS = 120
SIMILARITY_THRESHOLD = 0.85

langfuse_client = get_client()


def cosine_similarity(a, b) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


async def detect_implicit_negative(user_id: str, new_query: str) -> dict | None:
    """
    Looks at this user's single most recent prior request. If the new query is highly similar to it, within a short window, that's read as an implicit signal the PRIOR answer didn't satisfy them. Returns the prior request_log row if triggered, else None. Known false-positive case: a user re-asking the same question to double-check, not because the answer was wrong -- accepted here, same as any heuristic proxy, because nothing promoted from this signal skips human review.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        prior = await conn.fetchrow(
            """
            SELECT trace_id, query, answer FROM request_log WHERE user_id = $1 AND created_at > now() - ($2 * interval '1 second')
            ORDER BY created_at DESC LIMIT 1
            """,
            user_id,
            RETRY_WINDOW_SECONDS,
        )

    if prior is None or prior["answer"] is None:
        return None

    prior_vec = get_embedder().encode(f"search_query:{prior['query']}")
    new_vec = get_embedder().encode(f"search_query:{new_query}")
    similarity = cosine_similarity(prior_vec, new_vec)

    return dict(prior) if similarity >= SIMILARITY_THRESHOLD else None


async def record_implicit_negative(
    user_id: str, query: str, current_trace_id: str
) -> None:
    """Best-effort: never raises."""
    if user_id == "anonymous":
        return

    try:
        flagged_prior = await detect_implicit_negative(user_id, query)

        if not flagged_prior or flagged_prior["trace_id"] == current_trace_id:
            return

        langfuse_client.create_score(
            trace_id=flagged_prior["trace_id"],
            name="user.implicit_negative",
            value=1,
            data_type="BOOLEAN",
        )

        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                    INSERT INTO feedback (trace_id, query, answer, rating, reason, signal_type)
                    VALUES ($1, $2, $3, 'down','implicit: rapid similar rephrase', 'implicit_retry')
                    ON CONFLICT DO NOTHING
                """,
                flagged_prior["trace_id"],
                flagged_prior["query"],
                flagged_prior["answer"],
            )
    except Exception:
        print("implicit negative detection failed")
