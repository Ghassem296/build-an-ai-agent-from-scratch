from __future__ import annotations

import logging
import os
from typing import Any

from ddgs import DDGS
from mcp.server.fastmcp import FastMCP


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

mcp = FastMCP("web-search")


@mcp.tool()
def search_web(
    query: str,
    max_results: int = 5,
) -> list[dict[str, Any]]:
    """Search the web and return a list of relevant results.

    Args:
        query: Search query.
        max_results: Maximum number of results to return, from 1 to 10.
    """
    query = query.strip()

    if not query:
        raise ValueError("query cannot be empty")

    if not 1 <= max_results <= 10:
        raise ValueError("max_results must be between 1 and 10")

    try:
        with DDGS() as ddgs:
            results = ddgs.text(
                query,
                max_results=max_results,
            )
            return [
                {
                    "title": item.get("title", ""),
                    "url": item.get("href", ""),
                    "snippet": item.get("body", ""),
                }
                for item in results
            ]
    except Exception:
        logging.getLogger(__name__).exception("Web search failed")
        raise


if __name__ == "__main__":
    mcp.run(transport="stdio")