from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from ddgs import DDGS
from mcp.server.fastmcp import FastMCP


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

logger = logging.getLogger(__name__)

mcp = FastMCP("web-search")

# Indirection so tests can replace the real sleep with an instant stub.
_sleep = asyncio.sleep


def _search_sync(
    query: str,
    max_results: int,
) -> list[dict[str, Any]]:
    """Synchronous DDGS search (run inside a thread)."""
    with DDGS() as ddgs:
        results = list(
            ddgs.text(query, max_results=max_results)
        )

    return [
        {
            "title": item.get("title", ""),
            "url": item.get("href", ""),
            "snippet": item.get("body", ""),
        }
        for item in results
    ]


@mcp.tool()
async def search_web(
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

    max_attempts = 3
    backoff = 1.0

    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return await asyncio.to_thread(
                _search_sync,
                query,
                max_results,
            )
        except Exception as exc:
            last_error = exc
            logger.warning(
                "Web search attempt %d/%d failed: %s",
                attempt,
                max_attempts,
                exc,
            )
            if attempt < max_attempts:
                await _sleep(backoff)
                backoff *= 2

    logger.error(
        "Web search failed after %d attempts: %s",
        max_attempts,
        last_error,
    )
    raise RuntimeError(
        f"Web search failed after {max_attempts} attempts: {last_error}"
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")