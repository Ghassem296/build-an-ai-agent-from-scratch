"""Standalone demo entry point.

This mirrors what ``python -m app.main "<prompt>"`` does and exists for
readers following the book chapter by chapter. The supported entry point is
``app.main``; this module now reuses the shared MCP runtime instead of
carrying its own copy of the subprocess environment.

Usage::

    python mcp_agent.py
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from app.agent import Agent
from app.config import get_settings, require_api_key
from app.llm_client import LlmClient
from app.mcp_runtime import mcp_session


PROJECT_DIR = Path(__file__).resolve().parent

logger = logging.getLogger(__name__)

DEMO_PROMPT = (
    "Find the latest artificial intelligence news using web search, "
    "then summarize the three most relevant results. "
    "Include source URLs when available."
)


def create_llm_client() -> LlmClient:
    settings = get_settings()

    return LlmClient(
        model=settings.llm_model,
        api_base=settings.llm_api_base,
        api_key=settings.llm_api_key,
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout,
    )


async def main() -> None:
    """Connect to the MCP server and run the demo prompt."""
    settings = get_settings()

    require_api_key(settings)

    async with mcp_session(PROJECT_DIR) as session:
        tools_result = await session.list_tools()

        print("Connected to MCP server.")
        print("Available tools:")

        for tool in tools_result.tools:
            print(f"- {tool.name}: {tool.description or ''}")

        agent = Agent(
            llm_client=create_llm_client(),
            mcp_session=session,
            tool_timeout=settings.tool_timeout,
            max_context_chars=settings.agent_max_context_chars,
        )

        answer = await agent.run(
            DEMO_PROMPT,
            max_steps=settings.agent_max_steps,
        )

        print("\n=== FINAL ANSWER ===")
        print(answer)


if __name__ == "__main__":
    settings = get_settings()

    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExecution interrupted.")
    except Exception:
        logger.exception("Agent execution failed")
        raise
