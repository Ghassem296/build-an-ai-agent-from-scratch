from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from app.agent import Agent
from app.config import settings
from app.llm_client import LlmClient


def configure_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


async def run_agent(prompt: str) -> str:
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "tools.mcp_server"],
        env=os.environ.copy(),
    )

    async with AsyncExitStack() as stack:
        read_stream, write_stream = await stack.enter_async_context(
            stdio_client(server_params)
        )

        session = await stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )

        await session.initialize()

        llm_client = LlmClient(
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            api_base=settings.llm_api_base,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout,
        )

        agent = Agent(
            llm_client=llm_client,
            mcp_session=session,
            tool_timeout=settings.tool_timeout,
        )

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
        help="Prompt to send to the agent. If omitted, interactive mode starts.",
    )
    return parser.parse_args()


async def interactive_loop() -> None:
    while True:
        try:
            prompt = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if prompt.lower() in {"exit", "quit"}:
            break

        if not prompt:
            continue

        try:
            answer = await run_agent(prompt)
            print(f"\nAssistant: {answer}")
        except asyncio.CancelledError:
            raise
        except Exception:
            logging.getLogger(__name__).exception("Agent execution failed")
            print("خطا در اجرای درخواست؛ جزئیات در لاگ ثبت شده است.")


async def async_main() -> None:
    args = parse_args()
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


if __name__ == "__main__":
    main()