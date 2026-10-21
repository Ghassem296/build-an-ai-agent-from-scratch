from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from app.agent import Agent
from app.config import settings
from app.llm_client import LlmClient


# Project root: parent of the app directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def configure_logging() -> None:
    log_level = getattr(
        logging,
        str(settings.log_level).upper(),
        logging.INFO,
    )

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    # Enable diagnostics for LLM requests and HTTP retries.
    logging.getLogger("LiteLLM").setLevel(logging.DEBUG)
    logging.getLogger("litellm").setLevel(logging.DEBUG)
    logging.getLogger("openai").setLevel(logging.DEBUG)

    # INFO is usually enough to observe HTTP request activity.
    logging.getLogger("httpx").setLevel(logging.INFO)


def build_mcp_environment() -> dict[str, str]:
    """
    Pass only the environment variables required by the MCP server.

    Add other variables here only if tools.mcp_server explicitly needs them.
    """
    allowed_names = (
        "PATH",
        "SYSTEMROOT",
        "WINDIR",
        "TEMP",
        "TMP",
        "USERPROFILE",
        "APPDATA",
        "LOCALAPPDATA",
        "PYTHONIOENCODING",
        "PYTHONUTF8",
    )

    env = {
        name: os.environ[name]
        for name in allowed_names
        if name in os.environ
    }

    # Ensure the MCP subprocess uses UTF-8 where supported.
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    return env


async def run_agent(prompt: str) -> str:
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "tools.mcp_server"],
        cwd=str(PROJECT_ROOT),
        env=build_mcp_environment(),
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
        help=(
            "Prompt to send to the agent. "
            "If omitted, interactive mode starts."
        ),
    )
    return parser.parse_args()


async def interactive_loop() -> None:
    logger = logging.getLogger(__name__)

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
            logger.exception("Agent execution failed")
            print(
                "خطا در اجرای درخواست؛ "
                "جزئیات خطا در لاگ ثبت شده است."
            )


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
    except Exception:
        logging.getLogger(__name__).exception(
            "Application terminated unexpectedly"
        )
        raise


if __name__ == "__main__":
    main()