import operator
from typing import TypedDict, Annotated


class AgentState(TypedDict):
    task: str
    history: Annotated[
        list[str], operator.add
    ]  # accumulates -- without this, each node's return REPLACES history instead of appending to it.
    pending_action: str | None
    pending_arg: str | None
    final_answer: str | None
