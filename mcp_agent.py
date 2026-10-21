import asyncio
import logging
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from app.agent import Agent
from app.config import settings
from app.llm_client import LlmClient


PROJECT_DIR = Path(__file__).resolve().parent

logger = logging.getLogger(__name__)

AGENT_INSTRUCTIONS = (
    "You are a helpful research assistant. "
    "Use the provided tools when needed and summarize "
    "the results clearly. Distinguish verified facts from "
    "uncertain information."
)


def build_mcp_environment() -> dict[str, str]:
    """Pass only the environment variables required by the MCP server."""
    import os

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


async def main() -> None:
    """Connect to the MCP server and run the agent."""

    if not settings.llm_api_key:
        raise RuntimeError(
            "No API key found. Set LLM_API_KEY or XAI_1_API_KEY "
            "in the project's .env file."
        )

    llm_client = LlmClient(
        model=settings.llm_model,
        api_base=settings.llm_api_base,
        api_key=settings.llm_api_key,
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout,
    )

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "tools.mcp_server"],
        cwd=str(PROJECT_DIR),
        env=build_mcp_environment(),
    )

    async with stdio_client(server_params) as (
        read_stream,
        write_stream,
    ):
        async with ClientSession(
            read_stream,
            write_stream,
        ) as session:
            await session.initialize()

            tools_result = await session.list_tools()

            print("Connected to MCP server.")
            print("Available tools:")

            for tool in tools_result.tools:
                print(f"- {tool.name}: {tool.description or ''}")

            agent = Agent(
                llm_client=llm_client,
                mcp_session=session,
                tool_timeout=settings.tool_timeout,
            )

            prompt = (
                "Find the latest artificial intelligence news "
                "using web search, then summarize the three most "
                "relevant results. Include source URLs when available."
            )

            answer = await agent.run(
                prompt,
                max_steps=settings.agent_max_steps,
            )

            print("\n=== FINAL ANSWER ===")
            print(answer)


if __name__ == "__main__":
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