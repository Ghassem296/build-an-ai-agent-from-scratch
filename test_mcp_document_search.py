
from __future__ import annotations

import asyncio
from pathlib import Path

from app.mcp_runtime import mcp_session


async def main() -> None:
    project_root = Path(__file__).resolve().parent

    async with mcp_session(project_root) as session:
        response = await session.list_tools()

        tool_names = [tool.name for tool in response.tools]

        print("Available MCP tools:")
        for name in tool_names:
            print(f"- {name}")

        expected_tool = "search_documents_tool"

        if expected_tool not in tool_names:
            raise RuntimeError(
                f"MCP tool {expected_tool!r} was not registered."
            )

        print("\nCalling semantic document search...")

        result = await session.call_tool(
            expected_tool,
            {
                "query": "I forgot my password and cannot log in",
                "top_k": 2,
            },
        )

        if result.isError:
            raise RuntimeError(
                f"MCP tool returned an error: {result.content}"
            )

        for item in result.content:
            if hasattr(item, "text"):
                print(item.text)


if __name__ == "__main__":
    asyncio.run(main())
