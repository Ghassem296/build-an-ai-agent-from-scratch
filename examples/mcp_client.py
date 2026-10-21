import asyncio
import os
from pathlib import Path
from pprint import pprint

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


PROJECT_DIR = os.getenv(
    "AGENT_PROJECT_DIR",
    str(Path(__file__).resolve().parent),
)


def create_server_parameters() -> StdioServerParameters:
    """Create parameters for launching the MCP server."""

    return StdioServerParameters(
        command="uv",
        args=[
            "run",
            "--directory",
            PROJECT_DIR,
            "python",
            "-m",
            "tools.mcp_server",
        ],
    )


async def main() -> None:
    """Connect to the MCP server and test its tools."""

    server_params = create_server_parameters()

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

            print("========== AVAILABLE MCP TOOLS ==========")

            for tool in tools_result.tools:
                print(f"- {tool.name}: {tool.description or ''}")

            print("\n========== TEST SEARCH TOOL ==========")

            result = await session.call_tool(
                "search_web",
                {
                    "query": "latest artificial intelligence news",
                    "max_results": 3,
                },
            )

            print("\n========== RAW MCP RESULT ==========")
            pprint(result)

            print("\n========== STRUCTURED CONTENT ==========")
            pprint(
                getattr(result, "structuredContent", None)
            )

            print("\n========== CONTENT ITEMS ==========")

            for index, content in enumerate(
                result.content,
                start=1,
            ):
                print(f"\nItem {index}")
                print("Type:", getattr(content, "type", None))
                print("Text:", repr(getattr(content, "text", None)))

            print("\n========== ERROR STATUS ==========")
            print("isError:", result.isError)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nMCP client test interrupted.")