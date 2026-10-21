from agents.react import run_react
from agents.plan_and_execute import run_plan_and_execute

TASK = (
    "Check whether Pydantic's documentation mentions data validation. "
    "If it does, just confirm that and stop. "
    "If it does NOT, check FastAPI's documentation for a mention of validation instead."
)


def tool_calls(trace: list[dict]) -> int:
    return sum(1 for t in trace if "observation" in t)


if __name__ == "__main__":
    react_answer, react_trace = run_react(TASK)
    pae_answer, pae_trace = run_plan_and_execute(TASK)

    print(f"ReAct:            {tool_calls(react_trace)} tool calls -> {react_answer}")
    print(f"Plan-and-execute: {tool_calls(pae_trace)} tool calls -> {pae_answer}")
