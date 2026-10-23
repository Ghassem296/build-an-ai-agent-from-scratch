"""GAIA evaluation with the agent (book section 4.8).

Chapter 2 asked a bare LLM to solve GAIA problems and got low accuracy
because most of them need the web. Here the same problems are solved by the
agent, which can search through the MCP server, and the answer comes back as
a ``GaiaOutput`` object (structured output from section 4.7).

The problems are read from a local JSON or JSONL file, so this script needs
no extra dependency. Each problem must have the keys ``task_id``,
``Question`` and ``Final answer``.

Examples (a real run spends API tokens, so start with ``--limit 2``)::

    uv run python -m examples.gaia_agent_eval problems.jsonl --limit 2
    uv run python -m examples.gaia_agent_eval problems.jsonl --no-tools

``--no-tools`` runs the same agent without web search, which is the
baseline the book compares against.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncIterator

from pydantic import BaseModel

from app.agent import Agent
from app.config import get_settings
from app.execution_context import ExecutionContext
from app.llm_client import LlmClient, ToolCall
from app.mcp_runtime import mcp_session


PROJECT_ROOT = Path(__file__).resolve().parent.parent

REQUIRED_KEYS = ("task_id", "Question", "Final answer")


class GaiaOutput(BaseModel):
    is_solvable: bool
    unsolvable_reason: str = ""
    final_answer: str = ""


GAIA_PROMPT = (
    "You are a general AI assistant. You will be asked a question.\n"
    "Use the available tools when you need information you do not have.\n"
    "When you are done, call final_answer with:\n"
    "- is_solvable: true if you can answer, false if you cannot.\n"
    "- final_answer: your answer when solvable.\n"
    "- unsolvable_reason: why you cannot answer, when not solvable.\n\n"
    "Format of final_answer: a number, OR as few words as possible, OR a "
    "comma-separated list of numbers and/or strings.\n"
    "- Numbers: no thousands separators and no units ($, %) unless the "
    "question asks for them.\n"
    "- Strings: no articles, no abbreviations (for example city names), "
    "and spell digits out in words unless told otherwise.\n"
    "- Lists: apply the rules above to every element."
)


def is_correct(prediction: str | None, answer: str) -> bool:
    """GAIA exact match, ignoring case and surrounding whitespace."""
    if prediction is None:
        return False

    return prediction.strip().lower() == answer.strip().lower()


def load_problems(path: Path) -> list[dict[str, Any]]:
    """Read problems from a JSON array or a JSONL file."""
    text = path.read_text(encoding="utf-8").strip()

    if not text:
        raise ValueError(f"{path} is empty")

    if text.startswith("["):
        problems = json.loads(text)
    else:
        problems = [
            json.loads(line) for line in text.splitlines() if line.strip()
        ]

    for index, problem in enumerate(problems):
        missing = [key for key in REQUIRED_KEYS if key not in problem]

        if missing:
            raise ValueError(
                f"Problem #{index} is missing keys: {', '.join(missing)}"
            )

    return problems


class NoToolsSession:
    """Stand-in MCP session that exposes no tools (baseline run)."""

    async def list_tools(self) -> SimpleNamespace:
        return SimpleNamespace(tools=[])

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        raise RuntimeError(f"No tools are available (asked for {name!r})")


@asynccontextmanager
async def open_session(no_tools: bool) -> AsyncIterator[Any]:
    if no_tools:
        yield NoToolsSession()
        return

    async with mcp_session(PROJECT_ROOT) as session:
        yield session


def summarize_context(context: ExecutionContext) -> tuple[int, int]:
    """Return (llm steps, calls to real tools) for one execution."""
    steps = 0
    tool_calls = 0

    for event in context.get_events():
        if event.type != "assistant" or not isinstance(event.content, list):
            continue

        steps += 1
        tool_calls += sum(
            1
            for item in event.content
            if isinstance(item, ToolCall) and item.name != "final_answer"
        )

    return steps, tool_calls


async def evaluate_single(
    problem: dict[str, Any],
    *,
    client: LlmClient,
    session: Any,
    semaphore: asyncio.Semaphore,
    max_steps: int,
    tool_timeout: float,
) -> dict[str, Any]:
    """Run the agent on one problem and return a result record."""
    context = ExecutionContext()
    record: dict[str, Any] = {
        "task_id": problem["task_id"],
        "answer": problem["Final answer"],
    }

    async with semaphore:
        agent = Agent(
            client,
            session,
            tool_timeout=tool_timeout,
            system_prompt=GAIA_PROMPT,
            output_type=GaiaOutput,
        )

        try:
            result = await agent.run(
                problem["Question"],
                max_steps=max_steps,
                context=context,
            )
        except Exception as exc:  # keep the experiment going
            record.update(
                correct=False,
                is_solvable=None,
                prediction=None,
                error=str(exc),
            )
        else:
            if isinstance(result, GaiaOutput):
                record.update(
                    correct=is_correct(
                        result.final_answer, problem["Final answer"]
                    ),
                    is_solvable=result.is_solvable,
                    prediction=result.final_answer,
                    unsolvable_reason=result.unsolvable_reason,
                )
            else:
                # The step limit was reached and the model answered in text.
                record.update(
                    correct=False,
                    is_solvable=None,
                    prediction=None,
                    error="no structured answer (step limit reached)",
                    raw_answer=str(result)[:500],
                )

    record["steps"], record["tool_calls"] = summarize_context(context)

    return record


async def run_experiment(
    problems: list[dict[str, Any]],
    *,
    client: LlmClient,
    session: Any,
    concurrency: int,
    max_steps: int,
    tool_timeout: float,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(concurrency)
    results: list[dict[str, Any]] = []

    async def worker(problem: dict[str, Any]) -> None:
        record = await evaluate_single(
            problem,
            client=client,
            session=session,
            semaphore=semaphore,
            max_steps=max_steps,
            tool_timeout=tool_timeout,
        )
        results.append(record)
        if record["correct"]:
            mark = "OK "
        elif "error" in record:
            mark = "ERR"
        else:
            mark = "NO "
        print(
            f"[{len(results)}/{len(problems)}] {mark} {record['task_id']}"
            f" steps={record['steps']} tools={record['tool_calls']}",
            flush=True,
        )

    await asyncio.gather(*(worker(problem) for problem in problems))

    order = {p["task_id"]: i for i, p in enumerate(problems)}
    results.sort(key=lambda r: order[r["task_id"]])

    return results


def print_summary(results: list[dict[str, Any]], label: str) -> None:
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    unsolvable = sum(1 for r in results if r.get("is_solvable") is False)
    errors = sum(1 for r in results if "error" in r)
    avg_steps = sum(r["steps"] for r in results) / total if total else 0.0

    print(f"\n=== {label} ===")
    print(f"Accuracy:   {correct}/{total} ({correct / total:.0%})")
    print(f"Unsolvable: {unsolvable}")
    print(f"Errors:     {errors}")
    print(f"Avg steps:  {avg_steps:.1f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the agent on GAIA problems (section 4.8)."
    )
    parser.add_argument("problems", type=Path, help="JSON or JSONL file")
    parser.add_argument("--model", help="Override LLM_MODEL")
    parser.add_argument("--limit", type=int, help="Only the first N problems")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument(
        "--no-tools",
        action="store_true",
        help="Baseline: run without web search",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the per-problem results to this JSON file",
    )
    return parser.parse_args()


async def async_main(args: argparse.Namespace) -> None:
    settings = get_settings()
    problems = load_problems(args.problems)

    if args.limit is not None:
        problems = problems[: args.limit]

    client = LlmClient(
        model=args.model or settings.llm_model,
        api_key=settings.llm_api_key,
        api_base=settings.llm_api_base,
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout,
    )

    async with open_session(args.no_tools) as session:
        results = await run_experiment(
            problems,
            client=client,
            session=session,
            concurrency=args.concurrency,
            max_steps=args.max_steps,
            tool_timeout=settings.tool_timeout,
        )

    print_summary(results, "no tools" if args.no_tools else "with web search")

    if args.output:
        args.output.write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Results written to {args.output}")


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    args = parse_args()

    if args.concurrency < 1 or args.max_steps < 1:
        sys.exit("--concurrency and --max-steps must be at least 1")

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
