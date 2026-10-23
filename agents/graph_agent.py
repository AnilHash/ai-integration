from typing import Literal
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

from agents.graph_state import AgentState
from agents.react import (
    search_docs,
    word_count,
    ACTION_PATTERN,
    REACT_SYSTEM_PROMPT,
    TOOLS,
)
from app.llm_client import client, DEFAULT_MODEL


def reason_node(state: AgentState) -> dict:
    context = "\n".join(state["history"])
    response = (
        client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": REACT_SYSTEM_PROMPT},
                {"role": "user", "content": f"{state['task']}\n\n{context}"},
            ],
            temperature=0,
        )
        .choices[0]
        .message.content
    )
    if response is None:
        raise ValueError("Model returned no content")

    match = ACTION_PATTERN.search(response)
    if not match:
        return {"history": [response, "Observation: could not parse an Action"]}

    tool_name, arg = match.groups()
    if tool_name == "final_answer":
        return {"history": [response], "final_answer": arg}
    return {"history": [response], "pending_action": tool_name, "pending_arg": arg}


def human_review_node(state: AgentState) -> Command[Literal["act", "reason"]]:
    decision = interrupt(
        {
            "question": "Approve this tool call?",
            "action": state["pending_action"],
            "arg": state["pending_arg"],
        }
    )
    if decision:
        return Command(goto="act")
    return Command(
        goto="reason",
        update={"history": ["Observation: tool call rejected by human reviewer"]},
    )


def act_node(state: AgentState) -> dict:
    action = state["pending_action"]
    arg = state["pending_arg"]
    if action is None:
        observation = "no action specified"
    else:
        tool_fn = TOOLS.get(action)
        observation = str(tool_fn(arg or "")) if tool_fn else f"unknown tool '{action}'"
    return {"history": [f"Observation: {observation}"]}


def route_after_reason(state: AgentState) -> str:
    return END if state.get("final_answer") else "human_review"


graph = StateGraph(AgentState)
graph.add_node("reason", reason_node)
graph.add_node("human_review", human_review_node)
graph.add_node("act", act_node)

graph.add_edge(START, "reason")
graph.add_conditional_edges(
    "reason", route_after_reason, {"human_review": "human_review", END: END}
)
graph.add_edge(
    "act", "reason"
)  # after acting, go back to reasoning with the new observation

checkpointer = InMemorySaver()
compiled_graph = graph.compile(checkpointer=checkpointer)

config: RunnableConfig = {
    "configurable": {"thread_id": "research-run-1"},
    "recursion_limit": 15,
}
initial_state: AgentState = {
    "task": "What is vector database? Use search_docs to find out, then give a final answer.",
    "history": [],
    "pending_action": None,
    "pending_arg": None,
    "final_answer": None,
}

result = compiled_graph.invoke(initial_state, config)

if "__interrupt__" in result:
    payload = result["__interrupt__"][0].value
    print(f'Approval needed: {payload["action"]}("{payload["arg"]}")')
    result = compiled_graph.invoke(
        Command(resume=True), config
    )  # SAME config -- approve

print(result.get("final_answer"))

# config_2: RunnableConfig = {"configurable": {"thread_id": "research-run-2"}}
# result = compiled_graph.invoke(initial_state, config_2)

# if "__interrupt__" in result:
#     result = compiled_graph.invoke(Command(resume=False), config_2)  # reject this time

#     if (
#         "__interrupt__" in result
#     ):  # it looped back to reason, proposed another action, paused again
#         result = compiled_graph.invoke(
#             Command(resume=True), config_2
#         )  # approve the retry

# print(result.get("final_answer"))
