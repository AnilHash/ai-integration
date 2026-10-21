from dotenv import load_dotenv
from openai.types.chat import ChatCompletionMessageParam

load_dotenv()
from trace_logger import RunLogger

import re
from app.agent_tools import search_docs, word_count
from app.llm_client import client

TOOLS = {"search_docs": search_docs, "word_count": word_count}

REACT_SYSTEM_PROMPT = """Solve the task by interleaving Thought, Action, and Observation steps.

Available tools:
- search_docs(query): search the documentation, returns matching text
- word_count(text): counts words in a string
- final_answer(text): call this when you have enough information

Respond with EXACTLY this format for one step at a time:
Thought: <your reasoning> 
Action: <tool_name> (<argument in quotes>)

Stop after one Action and wait for the Observation."""


MODEL = "phi3:mini"
ACTION_PATTERN = re.compile(
    r'Action:\s*(\w+)\s*\("?(.*?)"?\)', re.DOTALL
)  # allows "tool (arg)" too


def run_react(task: str, max_steps: int = 6) -> tuple[str | None, list[dict]]:
    log = RunLogger(task, MODEL)
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": REACT_SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]
    trace = []

    for step in range(max_steps):
        response = (
            client.chat.completions.create(
                model=MODEL, messages=messages, temperature=0, max_tokens=200
            )
            .choices[0]
            .message.content
        )
        if response is None:
            log.step(step, "", error="model returned None")
            continue

        trace.append({"step": step, "raw": response})
        messages.append({"role": "assistant", "content": response})

        match = ACTION_PATTERN.search(response)
        if not match:
            err = "no parseable Action -- malformed output"
            trace[-1]["error"] = err
            log.step(step, response, error=err)
            messages.append(
                {
                    "role": "user",
                    "content": "Observation: could not parse an Action. Try again with the exact format.",
                }
            )
            continue

        tool_name, arg = match.groups()

        # final_answer was in your prompt but never handled -- this ends the loop
        if tool_name == "final_answer":
            log.step(step, response, observation="(final answer)")
            log.end("success", arg, steps_used=step + 1)
            return arg, trace

        if tool_name not in TOOLS:
            observation = f"unknown tool '{tool_name}'"
        else:
            observation = str(TOOLS[tool_name](arg))

        trace[-1]["observation"] = observation
        log.step(step, response, observation=observation)
        messages.append({"role": "user", "content": f"Observation: {observation}"})

    log.end("max_steps_exceeded", None, steps_used=max_steps)
    return None, trace
