from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from app.agent import Agent
from app.config import settings
from app.execution_context import ExecutionContext
from app.llm_client import LlmClient


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


def build_mcp_environment() -> dict[str, str]:
    """Pass only the environment variables required by the MCP server."""
    allowed_names = (
        "PATH",
        "SYSTEMROOT",
        "WINDIR",
        "TEMP",
        "TMP",
        "USERPROFILE",
        "APPDATA",
        "LOCALAPPDATA",
        "HOME",
        "LANG",
        "LC_ALL",
        "PYTHONIOENCODING",
        "PYTHONUTF8",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "no_proxy",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
    )

    env = {
        name: os.environ[name]
        for name in allowed_names
        if name in os.environ
    }

    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    return env


def create_llm_client() -> LlmClient:
    return LlmClient(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        api_base=settings.llm_api_base,
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout,
    )


@asynccontextmanager
async def mcp_session() -> AsyncIterator[ClientSession]:
    """Open an MCP session, yielding a ready ClientSession."""
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

        yield session


async def run_agent(prompt: str) -> str:
    async with mcp_session() as session:
        agent = Agent(
            llm_client=create_llm_client(),
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

    async with mcp_session() as session:
        agent = Agent(
            llm_client=create_llm_client(),
            mcp_session=session,
            tool_timeout=settings.tool_timeout,
        )

        session_context = ExecutionContext()
        agent.seed_system_prompt(session_context)

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