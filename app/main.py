from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from app.agent import SYSTEM_PROMPTS, Agent
from app.config import get_settings, require_api_key
from app.execution_context import ExecutionContext
from app.llm_client import LlmClient
from app.mcp_runtime import mcp_session


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def configure_logging() -> None:
    settings = get_settings()

    log_level = getattr(
        logging,
        str(settings.log_level).upper(),
        logging.INFO,
    )

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    if log_level == logging.DEBUG:
        logging.getLogger("LiteLLM").setLevel(logging.DEBUG)
        logging.getLogger("litellm").setLevel(logging.DEBUG)
        logging.getLogger("openai").setLevel(logging.DEBUG)
        logging.getLogger("httpx").setLevel(logging.INFO)
    else:
        logging.getLogger("LiteLLM").setLevel(logging.WARNING)
        logging.getLogger("litellm").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)


def create_llm_client() -> LlmClient:
    settings = get_settings()

    return LlmClient(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        api_base=settings.llm_api_base,
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout,
    )


def build_agent(session: object) -> Agent:
    settings = get_settings()

    return Agent(
        llm_client=create_llm_client(),
        mcp_session=session,
        tool_timeout=settings.tool_timeout,
        max_context_chars=settings.agent_max_context_chars,
        system_prompt=SYSTEM_PROMPTS[settings.system_prompt_profile],
    )


async def run_agent(prompt: str) -> str:
    settings = get_settings()

    async with mcp_session(PROJECT_ROOT) as session:
        agent = build_agent(session)

        return await agent.run(
            prompt,
            max_steps=settings.agent_max_steps,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the MCP-enabled LLM agent."
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help=(
            "Prompt to send to the agent. "
            "If omitted, interactive mode starts."
        ),
    )
    return parser.parse_args()


async def interactive_loop() -> None:
    logger = logging.getLogger(__name__)
    settings = get_settings()

    async with mcp_session(PROJECT_ROOT) as session:
        agent = build_agent(session)

        session_context = ExecutionContext()
        agent.seed_system_prompt(session_context)

        while True:
            try:
                # ``input()`` blocks the event loop; run it in a worker thread
                # so the MCP session keeps flushing stdio while we wait.
                prompt = (await asyncio.to_thread(input, "\nYou: ")).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if prompt.lower() in {"q", "exit", "quit"}:
                print("Goodbye!")
                break

            if not prompt:
                continue

            try:
                answer = await agent.run(
                    prompt,
                    max_steps=settings.agent_max_steps,
                    context=session_context,
                )
                print(f"\nAssistant: {answer}")
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Agent execution failed")
                print(
                    "Request failed. "
                    "See the log output for details."
                )


async def async_main() -> None:
    args = parse_args()
    require_api_key(get_settings())
    prompt = " ".join(args.prompt).strip()

    if prompt:
        answer = await run_agent(prompt)
        print(answer)
    else:
        await interactive_loop()


def main() -> None:
    configure_logging()

    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        pass
    except Exception:
        logging.getLogger(__name__).exception(
            "Application terminated unexpectedly"
        )
        raise


if __name__ == "__main__":
    main()
