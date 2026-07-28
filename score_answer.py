import re

from langfuse import get_client

langfuse_client = get_client()


def score_answer(trace_id: str, answer: str) -> dict:
    """
    Two independent heuristic checks, scored as separate boolean scores.
    Not a quality judgment - a mechanical check of one specific behaviour
    (citation presence) plus a sanity check (not degenerate output length).
    """
    cites_source = bool(re.search(r"doc-\d{3}", answer))
    reasonable_length = 20 <= len(answer) <= 600

    langfuse_client.create_score(
        trace_id=trace_id, name="cites_source", value=cites_source, data_type="BOOLEAN"
    )
    langfuse_client.create_score(
        trace_id=trace_id,
        name="reasonable_length",
        value=reasonable_length,
        data_type="BOOLEAN",
    )
    return {"cites_source": cites_source, "reasonable_length": reasonable_length}
