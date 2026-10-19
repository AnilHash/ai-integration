"""
Surfaces high-signal feedback for human review. Does NOT write to golden_50.json or behavioral_tests.json -- same principle as promote_baseline.py, applied to the least trustworthy data source in this pillar: uncontrolled user input.
"""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path


from dotenv import load_dotenv

load_dotenv()

from app.db import get_pool


CANDIDATES_PATH = Path(
    f"eval/golden/candidates_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.json"
)


async def main():
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
                SELECT id, trace_id, query, answer, signal_type, reason, created_at 
                FROM feedback
                WHERE rating = 'down' AND promoted = false
                ORDER BY created_at DESC 
            """
        )
    if not rows:
        print("No new negative feedback to review.")
        return

    candidates = [
        {
            "feedback_id": row["id"],
            "trace_id": row["trace_id"],
            "query": row["query"],
            "flagged_answer": row["answer"],
            "signal_type": row["signal_type"],
            "reason": row["reason"],
            "suggested_shape": "decide after reading: hallucination_guard / citation_accuracy / refusal_adversarial / in-scope golden_50 item",
        }
        for row in rows
    ]
    CANDIDATES_PATH.write_text(json.dumps(candidates, indent=2))
    print(f"{len(candidates)} candidate(s) written to {CANDIDATES_PATH}")
    print("Does NOT modify golden_50.json or behavioral_tests.json.")
    print("Read each one, decide the right shape, finish the fields that shape needs,")
    print("then copy the FINISHED item into the right file by hand.")

    async with pool.acquire() as conn:
        await conn.executemany(
            "UPDATE feedback SET promoted = true WHERE id = $1",
            [(r["id"],) for r in rows],
        )


if __name__ == "main":
    asyncio.run(main())
