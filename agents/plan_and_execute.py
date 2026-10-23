import re

from app.agent_tools import search_docs, word_count
from app.llm_client import client
from trace_logger import RunLogger

MODEL = "phi3:mini"

PLANNER_PROMPT = """
Given this task, write a numbered list of steps using ONLY these tools, in the exact order you'd execute them. Do not include conditional logic
("if X then Y") -- write a straight-line sequence of steps.

Tools: search_docs(query), word_count(text), final_answer(text)
Format: 
1. tool_name("argument")
2. tool_name("argument")
...
"""
TOOLS = {"search_docs": search_docs, "word_count": word_count}
STEP_PATTERN = re.compile(r'\d+\.\s*(\w+)\("?(.*?)"?\)')


def run_plan_and_execute(task: str) -> tuple[str | None, list[dict]]:
    log = RunLogger(task, MODEL, agent="plan_execute")

    plan_response = (
        client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": PLANNER_PROMPT},
                {"role": "user", "content": task},
            ],
            temperature=0,
            max_tokens=300,
        )
        .choices[0]
        .message.content
    )

    steps = STEP_PATTERN.findall(plan_response or "")
    log.plan(plan_response, steps)
    trace = [{"plan": plan_response, "parsed_steps": steps}]
    observations = []

    for i, (tool_name, arg) in enumerate(steps):
        if tool_name == "final_answer":
            # The planner wrote this BEFORE seeing any observations, so its text is a guess.
            # Stop executing and let the synthesis step build the answer from real data.
            log.tool_call(
                i,
                tool_name,
                arg,
                observation="(planned final_answer ignored; synthesizing)",
            )
            trace.append(
                {"step": tool_name, "arg": arg, "note": "ignored, synthesizing instead"}
            )
            break
        if tool_name not in TOOLS:
            err = "unknown tool in plan"
            trace.append({"step": tool_name, "error": err})
            log.tool_call(i, tool_name, arg, error=err)
            continue
        try:
            result = str(TOOLS[tool_name](arg))
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            trace.append({"step": tool_name, "arg": arg, "error": err})
            log.tool_call(i, tool_name, arg, error=err)
            continue
        observations.append(result)
        trace.append({"step": tool_name, "arg": arg, "observation": result})
        log.tool_call(i, tool_name, arg, observation=result)

    synthesis = (
        client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": f"Task: {task}\nGathered info: {observations}\nAnswer the task.",
                }
            ],
            temperature=0,
        )
        .choices[0]
        .message.content
    )
    trace.append({"step": "synthesis", "result": synthesis})
    log.synthesis(observations, synthesis)

    status = "success" if steps else "no_parseable_plan"
    log.end(status, synthesis, steps_used=len(steps))
    return synthesis, trace
