import json
import uuid
from datetime import datetime
from pathlib import Path


class RunLogger:
    """Appends one run's trace to a readable .log and a machine-readable .jsonl."""

    def __init__(
        self, task: str, model: str, agent: str = "react", log_dir: str = "traces"
    ):
        self.dir = Path(log_dir)
        self.dir.mkdir(exist_ok=True)
        self.txt_path = self.dir / "agent_traces.log"
        self.jsonl_path = self.dir / "agent_traces.jsonl"

        self.run_id = uuid.uuid4().hex[:8]
        self.task = task
        self.model = model
        self.agent = agent
        self.started = datetime.now()

        self._txt(
            f"\n{'=' * 80}\n"
            f"RUN {self.run_id} | {self.started:%Y-%m-%d %H:%M:%S} | agent={agent} | model={model}\n"
            f"TASK: {task}\n"
            f"{'=' * 80}\n"
        )
        self._jsonl({"event": "run_start", "task": task, "model": model})

    # --- internal writers (open/append/close each time so nothing is lost on crash)
    def _txt(self, text: str):
        with self.txt_path.open("a", encoding="utf-8") as f:
            f.write(text)

    def _jsonl(self, payload: dict):
        record = {
            "run_id": self.run_id,
            "agent": self.agent,
            "ts": datetime.now().isoformat(timespec="seconds"),
            **payload,
        }
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # --- ReAct-style step (model output + observation)
    def step(
        self,
        step: int,
        raw: str,
        observation: str | None = None,
        error: str | None = None,
    ):
        lines = [f"\n--- step {step} ---", f"[model]\n{raw.strip()}"]
        if error:
            lines.append(f"[error] {error}")
        if observation is not None:
            lines.append(f"[observation]\n{observation}")
        self._txt("\n".join(lines) + "\n")
        self._jsonl(
            {
                "event": "step",
                "step": step,
                "raw": raw,
                "observation": observation,
                "error": error,
            }
        )

    # --- Plan-and-execute pieces
    def plan(self, raw: str | None, parsed_steps: list[tuple[str, str]]):
        parsed = (
            "\n".join(f'  {i}. {t}("{a}")' for i, (t, a) in enumerate(parsed_steps))
            or "  (none)"
        )
        self._txt(
            f"\n--- plan ---\n[planner raw]\n{(raw or '').strip()}\n[parsed steps]\n{parsed}\n"
        )
        self._jsonl(
            {
                "event": "plan",
                "raw": raw,
                "parsed_steps": [{"tool": t, "arg": a} for t, a in parsed_steps],
            }
        )

    def tool_call(
        self,
        step: int,
        tool: str,
        arg: str,
        observation: str | None = None,
        error: str | None = None,
    ):
        lines = [f"\n--- step {step}: {tool}({arg!r}) ---"]
        if error:
            lines.append(f"[error] {error}")
        if observation is not None:
            lines.append(f"[observation]\n{observation}")
        self._txt("\n".join(lines) + "\n")
        self._jsonl(
            {
                "event": "tool_call",
                "step": step,
                "tool": tool,
                "arg": arg,
                "observation": observation,
                "error": error,
            }
        )

    def synthesis(self, observations: list[str], result: str | None):
        self._txt(
            f"\n--- synthesis ---\n[observations fed in]\n{observations}\n[model]\n{(result or '').strip()}\n"
        )
        self._jsonl(
            {"event": "synthesis", "observations": observations, "result": result}
        )

    def end(self, status: str, answer: str | None, steps_used: int):
        elapsed = (datetime.now() - self.started).total_seconds()
        self._txt(
            f"\n>>> RUN {self.run_id} END | status={status} | steps={steps_used} | {elapsed:.1f}s\n"
            f">>> ANSWER: {answer}\n"
        )
        self._jsonl(
            {
                "event": "run_end",
                "status": status,
                "answer": answer,
                "steps_used": steps_used,
                "elapsed_s": round(elapsed, 2),
            }
        )
