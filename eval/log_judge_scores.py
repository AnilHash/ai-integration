from langfuse import get_client

langfuse = get_client()


def log_judge_scores(trace_id: str, verdict) -> None:
    langfuse.create_score(
        trace_id=trace_id,
        name="judge.faithful",
        value=verdict.faithful,
        data_type="BOOLEAN",
    )
    langfuse.create_score(
        trace_id=trace_id,
        name="judge.relevant",
        value=verdict.relevant,
        data_type="BOOLEAN",
    )
    langfuse.create_score(
        trace_id=trace_id,
        name="judge.complete",
        value=verdict.complete,
        data_type="BOOLEAN",
    )
